//! Gaussian Mixture Models
//!
//! Provides implementation of GMMs using the EM algorithm.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::learning::gmm::{CovOption, GaussianMixtureModel};
//! use rusty_machine::learning::UnSupModel;
//!
//! let inputs = Matrix::new(4, 2, vec![1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 2.5]);
//! let test_inputs = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 2.9, 2.4, 2.5]);
//!
//! // Create gmm with k(=2) classes.
//! let mut model = GaussianMixtureModel::new(2);
//! model.set_max_iters(10);
//! model.cov_option = CovOption::Diagonal;
//!
//! // Where inputs is a Matrix with features in columns.
//! model.train(&inputs);
//!
//! // Where pred_data is a Matrix with features in columns.
//! let a = model.predict(&test_inputs);
//! println!("{:?}", a.data());
//! ```

use linalg::vector::Vector;
use linalg::matrix::Matrix;
use linalg::utils;

use learning::UnSupModel;
use learning::toolkit::rand_utils;

/// Covariance options for GMMs.
///
/// - Full : The full covariance structure.
/// - Regularized : Adds a regularization constant to the covariance diagonal.
/// - Diagonal : Only the diagonal covariance structure.
#[derive(Debug)]
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
    fn train(&mut self, inputs: &Matrix<f64>) {
        // Initialization:
        let k = self.comp_count;

        // TODO: Compute sample covariance of the data.
        // https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance
        let mut cov_vec = Vec::with_capacity(k);
        for _ in 0..k {
            cov_vec.push(Matrix::identity(inputs.cols()));
        }

        self.model_covars = Some(cov_vec);

        let random_rows: Vec<usize> = rand_utils::reservoir_sample(&(0..inputs.rows())
                                                                        .collect::<Vec<usize>>(),
                                                                   k);

        self.model_means = Some(inputs.select_rows(&random_rows));

        for _ in 0..self.max_iters {
            let log_lik_0 = self.log_lik;

            let (weights, log_lik_1) = self.membership_weights(inputs);

            if (log_lik_1 - log_lik_0).abs() < 1e-10 {
                break;
            }

            self.log_lik = log_lik_1;

            self.update_params(inputs, weights);
        }
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        if let (&Some(_), &Some(_)) = (&self.model_means, &self.model_covars) {
            self.membership_weights(inputs).0
        } else {
            panic!("Model has not been trained.");
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

    fn membership_weights(&self, inputs: &Matrix<f64>) -> (Matrix<f64>, f64) {
        let n = inputs.rows();

        let mut member_weights_data = Vec::with_capacity(n * self.comp_count);

        // We compute the determinants and inverses now
        let mut cov_sqrt_dets = Vec::with_capacity(self.comp_count);
        let mut cov_invs = Vec::with_capacity(self.comp_count);

        if let Some(ref covars) = self.model_covars {
            for cov in covars {
                // TODO: combine these. We compute det to get the inverse.
                let covar_det = cov.det();
                let covar_inv = cov.inverse();

                cov_sqrt_dets.push(covar_det.sqrt());
                cov_invs.push(covar_inv);
            }
        }

        let mut log_lik = 0f64;

        // Now we compute the membership weights
        if let Some(ref means) = self.model_means {
            for i in 0..n {
                let mut pdfs = Vec::with_capacity(self.comp_count);
                let x_i = inputs.select_rows(&[i]);

                for j in 0..self.comp_count {
                    let mu_j = means.select_rows(&[j]);
                    let diff = &x_i - mu_j;

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

        (Matrix::new(n, self.comp_count, member_weights_data),
         log_lik)
    }

    fn update_params(&mut self, inputs: &Matrix<f64>, membership_weights: Matrix<f64>) {
        let n = membership_weights.rows();
        let d = inputs.cols();

        let sum_weights = membership_weights.sum_rows();

        self.mix_weights = &sum_weights / (n as f64);

        let mut new_means = membership_weights.transpose() * inputs;

        for (idx, mean) in new_means.mut_data().chunks_mut(d).enumerate() {
            for m in mean {
                *m = *m / sum_weights[idx];
            }
        }

        let mut new_covs = Vec::with_capacity(self.comp_count);

        for k in 0..self.comp_count {
            let mut cov_mat = Matrix::zeros(d, d);

            for i in 0..n {
                let diff = inputs.select_rows(&[i]) - new_means.select_rows(&[k]);
                cov_mat = cov_mat + self.compute_cov(diff, membership_weights[[i, k]]);
            }
            new_covs.push(cov_mat / sum_weights[k]);

        }

        self.model_means = Some(new_means);
        self.model_covars = Some(new_covs);
    }

    fn compute_cov(&self, diff: Matrix<f64>, weight: f64) -> Matrix<f64> {
        match self.cov_option {
            CovOption::Full => (diff.transpose() * diff) * weight,
            CovOption::Regularized(eps) => (diff.transpose() * diff) * weight + eps,
            CovOption::Diagonal => Matrix::from_diag(&diff.elemul(&diff).into_vec()),
        }
    }
}
