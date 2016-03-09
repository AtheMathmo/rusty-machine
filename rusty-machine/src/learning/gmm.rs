use linalg::vector::Vector;
use linalg::matrix::Matrix;

use learning::UnSupModel;

pub const SQRT_2_PI: f64 = 2.50662827463100050241576528481104525_f64;

/// A gaussian mixture model
///
/// Currently contains it's own parameters.
/// In future we should construct a finite
/// mixture model and have the GaussianMixtureModel
/// own this.
pub struct GaussianMixtureModel {
    comp_count: usize,
    mix_weights: Vector<f64>,
    model_means: Matrix<f64>,
    model_covars: Vec<Matrix<f64>>,
}

impl UnSupModel<Matrix<f64>, Matrix<f64>> for GaussianMixtureModel {
    /// Train the model using inputs.
    fn train(&mut self, inputs: &Matrix<f64>) {
    	// Need to do initialization

    	for _ in 0..10 {
    		let weights = self.membership_weights(inputs);
    		self.update_params(inputs, weights);
    	}
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        self.membership_weights(inputs)
    }
}

impl GaussianMixtureModel {
    fn membership_weights(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        let d = self.model_means.cols();
        let n = inputs.rows();

        let pi_d = SQRT_2_PI.powi(d as i32);

        let mut member_weights_data = Vec::with_capacity(n * self.comp_count);

        // We compute the determinants and inverses now
        let mut cov_sqrt_dets = Vec::with_capacity(self.comp_count);
        let mut cov_invs = Vec::with_capacity(self.comp_count);

        for k in 0..self.comp_count {
            // TODO: combine these. We compute det to get the inverse.
            let covar_det = self.model_covars[k].det();
            let covar_inv = self.model_covars[k].inverse();

            cov_sqrt_dets.push(covar_det.sqrt());
            cov_invs.push(covar_inv);
        }

        // Now we compute the membership weights
        for i in 0..n {
            for k in 0..self.comp_count {

                let mut pdf_inv_sum = 0f64;
                let x_i = inputs.select_rows(&[i]);

                for j in 0..self.comp_count {
                    if j == k {
                        continue;
                    }

                    let mu_j = self.model_means.select_rows(&[j]);
                    let diff = &x_i - mu_j;

                    // diff is 1xd
                    let exponent = (&diff * &cov_invs[j] * diff.transpose() * 0.5).into_vec()[0];

                    pdf_inv_sum += exponent.exp() * cov_sqrt_dets[j] / self.mix_weights[j];
                }

                member_weights_data.push(self.mix_weights[k] * pi_d * pdf_inv_sum);
            }
        }

        Matrix::new(n, self.comp_count, member_weights_data)
    }

    fn update_params(&mut self, inputs: &Matrix<f64>, membership_weights: Matrix<f64>) {
        let n = membership_weights.rows();
        let d = membership_weights.cols();

        let sum_weights = membership_weights.sum_rows();

        self.mix_weights = &sum_weights / (n as f64);

        let mut new_means = membership_weights.transpose() * inputs;

        for (idx, mean) in new_means.mut_data().chunks_mut(d).enumerate() {
            for i in 0..d {
                mean[i] = mean[i] / sum_weights[idx];
            }
        }

        self.model_means = new_means;

        for k in 0..self.comp_count {
            let mut cov_mat = Matrix::zeros(d, d);

            for i in 0..n {
                let diff = inputs.select_rows(&[i]) - self.model_means.select_rows(&[k]);
                cov_mat = cov_mat + (diff.transpose() * diff) * membership_weights[[i, k]];
            }

            self.model_covars[k] = cov_mat / sum_weights[k];
        }

    }
}
