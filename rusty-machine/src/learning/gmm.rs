//! Gaussian Mixture Models
//!
//! Provides implementation of GMMs.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::learning::gmm::GaussianMixtureModel;
//! use rusty_machine::learning::UnSupModel;
//!
//! let inputs = Matrix::new(4, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 5.0, 1.0]);
//! let test_inputs = Matrix::new(2, 2, vec![1.0, 3.5, 4.0, 1.0]);
//!
//! // Create gmm with k(=2) classes.
//! let mut model = GaussianMixtureModel::new(3);
//!
//! // Where inputs is a Matrix with features in columns.
//! model.train(&inputs);
//!
//! // Where pred_data is a Matrix with features in columns.
//! let a = model.predict(&test_inputs);
//! assert!(false);
//! ```

use linalg::vector::Vector;
use linalg::matrix::Matrix;
use linalg::utils;

use learning::UnSupModel;

/// A Gaussian Micture Model
///
/// Currently contains it's own parameters.
/// In future we should construct a finite
/// mixture model and have the GaussianMixtureModel
/// own this.
pub struct GaussianMixtureModel {
    comp_count: usize,
    mix_weights: Vector<f64>,
    model_means: Option<Matrix<f64>>,
    model_covars: Option<Vec<Matrix<f64>>>,
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

        // TODO: Use randomization, or k-means.
        self.model_means = Some(inputs.select_rows(&(0..k).collect::<Vec<usize>>()[..]));

        for _ in 0..4 {
            let weights = self.membership_weights(inputs);
            self.update_params(inputs, weights);
        }
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        if let (&Some(_), &Some(_)) = (&self.model_means, &self.model_covars) {
            self.membership_weights(inputs)
        } else {
            panic!("Model has not been trained.");
        }

    }
}

impl GaussianMixtureModel {
    /// Constructs a new Gaussian Mixture Model
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
        }
    }

    fn membership_weights(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        let n = inputs.rows();

        let mut member_weights_data = Vec::with_capacity(n * self.comp_count);

        // We compute the determinants and inverses now
        let mut cov_sqrt_dets = Vec::with_capacity(self.comp_count);
        let mut cov_invs = Vec::with_capacity(self.comp_count);

        if let Some(ref covars) = self.model_covars {
            for k in 0..self.comp_count {
                // TODO: combine these. We compute det to get the inverse.
                let covar_det = covars[k].det();
                let covar_inv = covars[k].inverse();

                cov_sqrt_dets.push(covar_det.sqrt());
                cov_invs.push(covar_inv);
            }
        }

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

                for j in 0..self.comp_count {
                    let weighted_pdf_sum = utils::dot(&pdfs, self.mix_weights.data());
                    member_weights_data.push(self.mix_weights[j] * pdfs[j] /
                                             (weighted_pdf_sum));
                }


            }
        }

        Matrix::new(n, self.comp_count, member_weights_data)
    }

    fn update_params(&mut self, inputs: &Matrix<f64>, membership_weights: Matrix<f64>) {
        let n = membership_weights.rows();
        let d = inputs.cols();

        let sum_weights = membership_weights.sum_rows();

        self.mix_weights = &sum_weights / (n as f64);

        let mut new_means = membership_weights.transpose() * inputs;

        // TODO: Optimize
        for (idx, mean) in new_means.mut_data().chunks_mut(d).enumerate() {
            for i in 0..d {
                mean[i] = mean[i] / sum_weights[idx];
            }
        }

        let mut new_covs = Vec::with_capacity(self.comp_count);

        // TODO: Optimize
        for k in 0..self.comp_count {
            let mut cov_mat = Matrix::zeros(d, d);

            for i in 0..n {
                let diff = inputs.select_rows(&[i]) - new_means.select_rows(&[k]);
                cov_mat = cov_mat + (diff.transpose() * diff) * membership_weights[[i, k]];
            }
            new_covs.push(cov_mat / sum_weights[k]);

        }

        self.model_means = Some(new_means);
        self.model_covars = Some(new_covs);
    }
}
