//! Gaussian Mixture Models
//!
//! Provides implementation of GMMs using the EM algorithm.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::learning::gmm::{CovOption, GaussianMixtureModel};
//! use rusty_machine::learning::UnSupModel;
//!
//! let inputs = Matrix::new(4, 2, vec![1.0, 2.0, -3.0, -3.0, 0.1, 1.5, -5.0, -2.5]);
//! let test_inputs = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 2.9, -4.4, -2.5]);
//!
//! // Create gmm with k(=2) classes.
//! let mut model = GaussianMixtureModel::new(2);
//! model.set_max_iters(10);
//! model.cov_option = CovOption::Diagonal;
//!
//! // Where inputs is a Matrix with features in columns.
//! model.train(&inputs).unwrap();
//!
//! // Print the means and covariances of the GMM
//! println!("{:?}", model.means());
//! println!("{:?}", model.covariances());
//!
//! // Where test_inputs is a Matrix with features in columns.
//! let post_probs = model.predict(&test_inputs).unwrap();
//!
//! // Probabilities that each point comes from each Gaussian.
//! println!("{:?}", post_probs.data());
//! ```
use linalg::{Matrix, MatrixSlice, Vector, BaseMatrix, BaseMatrixMut, Axes};
use rulinalg::utils;
use rulinalg::matrix::decomposition::{PartialPivLu};

use learning::{LearningResult, UnSupModel};
use learning::toolkit::rand_utils;
use learning::error::{Error, ErrorKind};

/// Covariance options for GMMs.
///
/// - Full : The full covariance structure.
/// - Regularized : Adds a regularization constant to the covariance diagonal.
/// - Diagonal : Only the diagonal covariance structure.
#[derive(Clone, Copy, Debug)]
pub enum CovOption {
    /// The full covariance structure.
    Full,
    /// Adds a regularization constant to the covariance diagonal.
    Regularized(f64),
    /// Only the diagonal covariance structure.
    Diagonal,
}


/// A Gaussian Mixture Model
#[derive(Debug)]
pub struct GaussianMixtureModel {
    comp_count: usize,
    mix_weights: Vector<f64>,
    model_means: Option<Matrix<f64>>,
    model_covars: Option<Vec<Matrix<f64>>>,
    log_lik: f64,
    max_iters: usize,
    /// The covariance options for the GMM.
    pub cov_option: CovOption,
}

impl UnSupModel<Matrix<f64>, Matrix<f64>> for GaussianMixtureModel {
    /// Train the model using inputs.
    fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        let reg_value = if inputs.rows() > 1 {
            1f64 / (inputs.rows() - 1) as f64
        } else {
            return Err(Error::new(ErrorKind::InvalidData, "Only one row of data provided."));
        };

        // Initialization:
        let k = self.comp_count;

        self.model_covars = {
            let cov_mat = self.initialize_covariances(inputs, reg_value)?;
            Some(vec![cov_mat; k])
        };

        let random_rows: Vec<usize> =
            rand_utils::reservoir_sample(&(0..inputs.rows()).collect::<Vec<usize>>(), k);
        self.model_means = Some(inputs.select_rows(&random_rows));

        for _ in 0..self.max_iters {
            let log_lik_0 = self.log_lik;

            let (weights, log_lik_1) = self.membership_weights(inputs)?;

            if (log_lik_1 - log_lik_0).abs() < 1e-15 {
                break;
            }

            self.log_lik = log_lik_1;

            self.update_params(inputs, weights);
        }

        Ok(())
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        if let (&Some(_), &Some(_)) = (&self.model_means, &self.model_covars) {
            Ok(self.membership_weights(inputs)?.0)
        } else {
            Err(Error::new_untrained())
        }

    }
}

impl GaussianMixtureModel {
    /// Constructs a new Gaussian Mixture Model
    ///
    /// Defaults to 100 maximum iterations and
    /// full covariance structure.
    ///
    /// # Examples
    /// ```
    /// use rusty_machine::learning::gmm::GaussianMixtureModel;
    ///
    /// let gmm = GaussianMixtureModel::new(3);
    /// ```
    pub fn new(k: usize) -> GaussianMixtureModel {
        GaussianMixtureModel {
            comp_count: k,
            mix_weights: Vector::ones(k) / (k as f64),
            model_means: None,
            model_covars: None,
            log_lik: 0f64,
            max_iters: 100,
            cov_option: CovOption::Full,
        }
    }

    /// Constructs a new GMM with the specified prior mixture weights.
    ///
    /// The mixture weights must have the same length as the number of components.
    /// Each element of the mixture weights must be non-negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gmm::GaussianMixtureModel;
    /// use rusty_machine::linalg::Vector;
    ///
    /// let mix_weights = Vector::new(vec![0.25, 0.25, 0.5]);
    ///
    /// let gmm = GaussianMixtureModel::with_weights(3, mix_weights).unwrap();
    /// ```
    ///
    /// # Failures
    ///
    /// Fails if either of the following conditions are met:
    ///
    /// - Mixture weights do not have length k.
    /// - Mixture weights have a negative entry.
    pub fn with_weights(k: usize, mixture_weights: Vector<f64>) -> LearningResult<GaussianMixtureModel> {
        if mixture_weights.size() != k {
            Err(Error::new(ErrorKind::InvalidParameters, "Mixture weights must have length k."))
        } else if mixture_weights.data().iter().any(|&x| x < 0f64) {
            Err(Error::new(ErrorKind::InvalidParameters, "Mixture weights must have only non-negative entries."))
        } else {
            let sum = mixture_weights.sum();
            let normalized_weights = mixture_weights / sum;

            Ok(GaussianMixtureModel {
                comp_count: k,
                mix_weights: normalized_weights,
                model_means: None,
                model_covars: None,
                log_lik: 0f64,
                max_iters: 100,
                cov_option: CovOption::Full,
            })
        }
    }

    /// The model means
    ///
    /// Returns an Option<&Matrix<f64>> containing
    /// the model means. Each row represents
    /// the mean of one of the Gaussians.
    pub fn means(&self) -> Option<&Matrix<f64>> {
        self.model_means.as_ref()
    }

    /// The model covariances
    ///
    /// Returns an Option<&Vec<Matrix<f64>>> containing
    /// the model covariances. Each Matrix in the vector
    /// is the covariance of one of the Gaussians.
    pub fn covariances(&self) -> Option<&Vec<Matrix<f64>>> {
        self.model_covars.as_ref()
    }

    /// The model mixture weights
    ///
    /// Returns a reference to the model mixture weights.
    /// These are the weighted contributions of each underlying
    /// Gaussian to the model distribution.
    pub fn mixture_weights(&self) -> &Vector<f64> {
        &self.mix_weights
    }

    /// Sets the max number of iterations for the EM algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gmm::GaussianMixtureModel;
    ///
    /// let mut gmm = GaussianMixtureModel::new(2);
    /// gmm.set_max_iters(5);
    /// ```
    pub fn set_max_iters(&mut self, iters: usize) {
        self.max_iters = iters;
    }

    fn initialize_covariances(&self, inputs: &Matrix<f64>, reg_value: f64) -> LearningResult<Matrix<f64>> {
        match self.cov_option {
            CovOption::Diagonal => {
                let variance = inputs.variance(Axes::Row)?;
                Ok(Matrix::from_diag(variance.data()) * reg_value.sqrt())
            }

            CovOption::Full | CovOption::Regularized(_) => {
                let means = inputs.mean(Axes::Row);
                let mut cov_mat = Matrix::zeros(inputs.cols(), inputs.cols());
                for (j, mut row) in cov_mat.row_iter_mut().enumerate() {
                    for (k, elem) in row.iter_mut().enumerate() {
                        *elem = inputs.row_iter().map(|r| {
                            (r[j] - means[j]) * (r[k] - means[k])
                        }).sum::<f64>();
                    }
                }
                cov_mat *= reg_value;
                if let CovOption::Regularized(eps) = self.cov_option {
                    cov_mat += Matrix::<f64>::identity(cov_mat.cols()) * eps;
                }
                Ok(cov_mat)
            }
        }
    }

    fn membership_weights(&self, inputs: &Matrix<f64>) -> LearningResult<(Matrix<f64>, f64)> {
        let n = inputs.rows();

        let mut member_weights_data = Vec::with_capacity(n * self.comp_count);

        // We compute the determinants and inverses now
        let mut cov_sqrt_dets = Vec::with_capacity(self.comp_count);
        let mut cov_invs = Vec::with_capacity(self.comp_count);

        if let Some(ref covars) = self.model_covars {
            for cov in covars {
                let lup = PartialPivLu::decompose(cov.clone()).expect("Covariance could not be lup decomposed");
                let covar_det = lup.det();
                // TODO: We can probably remove this inverse for a more stable solve elsewhere.
                let covar_inv = lup.inverse().map_err(Error::from)?;

                cov_sqrt_dets.push(covar_det.sqrt());
                cov_invs.push(covar_inv);
            }
        }

        let mut log_lik = 0f64;

        // Now we compute the membership weights
        if let Some(ref means) = self.model_means {
            for i in 0..n {
                let mut pdfs = Vec::with_capacity(self.comp_count);
                let x_i = MatrixSlice::from_matrix(inputs, [i, 0], 1, inputs.cols());

                for j in 0..self.comp_count {
                    let mu_j = MatrixSlice::from_matrix(means, [j, 0], 1, means.cols());
                    let diff = x_i - mu_j;

                    let pdf = (&diff * &cov_invs[j] * diff.transpose() * -0.5).into_vec()[0]
                        .exp() / cov_sqrt_dets[j];
                    pdfs.push(pdf);
                }

                let weighted_pdf_sum = utils::dot(&pdfs, self.mix_weights.data());

                for (idx, pdf) in pdfs.iter().enumerate() {
                    member_weights_data.push(self.mix_weights[idx] * pdf / (weighted_pdf_sum));
                }

                log_lik += weighted_pdf_sum.ln();
            }
        }

        Ok((Matrix::new(n, self.comp_count, member_weights_data), log_lik))
    }

    fn update_params(&mut self, inputs: &Matrix<f64>, membership_weights: Matrix<f64>) {
        let n = membership_weights.rows();
        let d = inputs.cols();

        let sum_weights = membership_weights.sum_rows();

        self.mix_weights = &sum_weights / (n as f64);

        let mut new_means = membership_weights.transpose() * inputs;

        for (mut mean, w) in new_means.row_iter_mut().zip(sum_weights.data().iter()) {
            *mean /= *w;
        }

        let mut new_covs = Vec::with_capacity(self.comp_count);

        for k in 0..self.comp_count {
            let mut cov_mat = Matrix::zeros(d, d);
            let new_means_k = MatrixSlice::from_matrix(&new_means, [k, 0], 1, d);

            for i in 0..n {
                let inputs_i = MatrixSlice::from_matrix(inputs, [i, 0], 1, d);
                let diff = inputs_i - new_means_k;
                cov_mat += self.compute_cov(diff, membership_weights[[i, k]]);
            }

            if let CovOption::Regularized(eps) = self.cov_option {
                cov_mat += Matrix::<f64>::identity(cov_mat.cols()) * eps;
            }

            new_covs.push(cov_mat / sum_weights[k]);

        }

        self.model_means = Some(new_means);
        self.model_covars = Some(new_covs);
    }

    fn compute_cov(&self, diff: Matrix<f64>, weight: f64) -> Matrix<f64> {
        match self.cov_option {
            CovOption::Full | CovOption::Regularized(_) => (diff.transpose() * diff) * weight,
            CovOption::Diagonal => Matrix::from_diag(&diff.elemul(&diff).into_vec()) * weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianMixtureModel;
    use linalg::Vector;

    #[test]
    fn test_means_none() {
        let model = GaussianMixtureModel::new(5);

        assert_eq!(model.means(), None);
    }

    #[test]
    fn test_covars_none() {
        let model = GaussianMixtureModel::new(5);

        assert_eq!(model.covariances(), None);
    }

    #[test]
    fn test_negative_mixtures() {
        let mix_weights = Vector::new(vec![-0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }

    #[test]
    fn test_wrong_length_mixtures() {
        let mix_weights = Vector::new(vec![0.1, 0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }
}
