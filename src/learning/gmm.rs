//! Gaussian Mixture Models
//!
//! Provides implementation of GMMs using the EM algorithm.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::learning::gmm::{CovOption, GaussianMixtureModel, Random, CholeskyFull, Diagonal};
//! use rusty_machine::learning::UnSupModel;
//!
//! let inputs = Matrix::new(4, 2, vec![1.0, 2.0, -3.0, -3.0, 0.1, 1.5, -5.0, -2.5]);
//! let test_inputs = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 2.9, -4.4, -2.5]);
//!
//! // Create gmm with k(=2) classes.
//! let mut model: GaussianMixtureModel<Random, Diagonal> = GaussianMixtureModel::new(2);
//! model.set_max_iters(10);
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
extern crate rand;

use linalg::{Matrix, Vector, BaseMatrix, BaseMatrixMut, Axes};

use learning::{LearningResult, UnSupModel};
use learning::error::{Error, ErrorKind};

use std::f64::consts::PI;
use std::f64::EPSILON;
use std::mem;
use std::marker::PhantomData;

const CONVERGENCE_EPS: f64 = 1.0e-15;

/// A Gaussian Mixture Model
#[derive(Debug)]
pub struct GaussianMixtureModel<T: Initializer, C: CovOption<T>> {
    comp_count: usize,
    // [n_features]
    mix_weights: Vector<f64>,
    // [n_components, n_features]
    model_means: Option<Matrix<f64>>,
    // n_components elements: [n_features, n_features]
    model_covars: Option<Vec<Matrix<f64>>>,
    // n_components elements: [n_features, n_features]
    precisions: Option<Vec<Matrix<f64>>>,
    log_lik: f64,
    max_iters: usize,
    phantom: PhantomData<T>,
    /// Struct that calculates covariance and gaussian parameters
    pub cov_option: C,
}

impl<T, C> UnSupModel<Matrix<f64>, Matrix<f64>> for GaussianMixtureModel<T, C> 
    where T: Initializer,
          C: CovOption<T> + Default
{
    /// Train the model using inputs.
    fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        // Move k to stack to appease the borrow checker 
        let k = self.comp_count;

        // Initialize empty matrices for covariances and means
        self.model_covars = Some(
            vec![Matrix::zeros(inputs.cols(), inputs.cols()); k]);
        self.model_means = Some(Matrix::zeros(inputs.cols(), k));

        // Initialize responsibitilies and calculate parameters
        self.update_gaussian_parameters(inputs, try!(T::init_resp(k, inputs)));
        self.precisions =
            Some(try!(self.cov_option.compute_precision(
                        self.model_covars.as_ref().unwrap())));

        self.log_lik = 0.;

        for _ in 0..self.max_iters {
            let log_lik_0 = self.log_lik;

            // e_step
            let (log_prob_norm, mut resp) = self.estimate_log_prob_resp(inputs);
            // Return to normal space
            for v in resp.iter_mut() { *v = v.exp(); }

            // m_step
            self.update_gaussian_parameters(inputs, resp);
            self.precisions =
                Some(try!(self.cov_option.compute_precision(
                            self.model_covars.as_ref().unwrap())));
            
            // check for convergence
            let log_lik_1 = log_prob_norm.mean();
            if (log_lik_0 - log_lik_1).abs() < CONVERGENCE_EPS {
                break;
            }

            self.log_lik = log_lik_1;
        }

        Ok(())
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        if let (&Some(_), &Some(_)) = (&self.model_means, &self.model_covars) {
            Ok(self.estimate_weighted_log_prob(inputs))
        } else {
            Err(Error::new_untrained())
        }

    }
}

impl<T, C> GaussianMixtureModel<T, C> 
    where T: Initializer,
          C: CovOption<T> + Default
{
    /// Constructs a new Gaussian Mixture Model
    ///
    /// Defaults to 100 maximum iterations and
    /// full covariance structure.
    ///
    /// # Examples
    /// ```
    /// use rusty_machine::learning::gmm::{GaussianMixtureModel, Random, CholeskyFull};
    ///
    /// let gmm: GaussianMixtureModel<Random, CholeskyFull> = GaussianMixtureModel::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        Self::with_weights(k, Vector::ones(k) / k as f64).unwrap()
    }

    /// Constructs a new GMM with the specified prior mixture weights.
    ///
    /// The mixture weights must have the same length as the number of components.
    /// Each element of the mixture weights must be non-negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gmm::{GaussianMixtureModel, Random, CholeskyFull};
    /// use rusty_machine::linalg::Vector;
    ///
    /// let mix_weights = Vector::new(vec![0.25, 0.25, 0.5]);
    ///
    /// let gmm: GaussianMixtureModel<Random, CholeskyFull> = GaussianMixtureModel::with_weights(3, mix_weights).unwrap();
    /// ```
    ///
    /// # Failures
    ///
    /// Fails if either of the following conditions are met:
    ///
    /// - Mixture weights do not have length k.
    /// - Mixture weights have a negative entry.
    pub fn with_weights(k: usize, mixture_weights: Vector<f64>) -> LearningResult<Self> {
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
                precisions: None,
                log_lik: 0.,
                max_iters: 100,
                cov_option: C::default(),
                phantom: PhantomData,
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

    /// The model mean likelihood
    ///
    /// Returns the mean log likelihood of the model
    pub fn log_lik(&self) -> f64 {
        self.log_lik
    }

    /// Sets the max number of iterations for the EM algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gmm::{GaussianMixtureModel, Random, CholeskyFull};
    ///
    /// let mut gmm: GaussianMixtureModel<Random, CholeskyFull> = GaussianMixtureModel::new(2);
    /// gmm.set_max_iters(5);
    /// ```
    pub fn set_max_iters(&mut self, iters: usize) {
        self.max_iters = iters;
    }

    // == Parameters
    // inputs : [n_samples, n_features]
    //
    // == Returns
    // weighted_log_prob : [n_features, n_components]
    //
    fn estimate_weighted_log_prob(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        let mut log_prob = self.estimate_log_prob(&inputs);
        // println!("log_prob: \n{:.4}", log_prob.select_rows(&[0, 1, 2, 3, 4, 5, 6]));
        // println!("mix_weights: \n{:?}", &self.mix_weights);
        let log_weights = self.mix_weights.iter().map(|w| w.ln());
        for (lp, lw) in log_prob.iter_mut().zip(log_weights) {
            *lp += lw;
        }
        log_prob
    }

    // called estimate_log_gaussian_prob in scipy
    //
    // == Paramers
    // inputs : [n_samples, n_features]
    //
    //
    fn estimate_log_prob(&self, inputs: &Matrix<f64>) -> Matrix<f64> {
        let precisions: &Vec<Matrix<f64>> = self.precisions.as_ref().unwrap();
        let ref model_means = self.model_means.as_ref().unwrap();
        // The log of the determinant for each precision matrix
        let log_det = precisions.iter().map(|m| m.det().ln());
        // println!("log_det: {:?}", log_det);

        let mut log_prob = Matrix::zeros(inputs.rows(), self.comp_count);
        for k in 0..self.comp_count {
            let prec = &precisions[k];
            // y is a matrix of shape [n_samples, n_features]
            let mut y = inputs * prec;
            // println!("y: \n{:.4}", &y.select_rows(&[0, 1, 2, 3, 4, 5, 6]));
            // Matrix of shape [1, n_features]
            let z: Matrix<f64> = model_means.select_rows(&[k]) * prec;
            // println!("z: \n{:.4}", &z);
            
            // Subtract the mean of each column from the matrix y
            for col in 0..y.cols() {
                for row in 0..y.rows() {
                    y[[row, col]] -= z[[0, col]];
                }
            }
            // println!("y: \n{:.4}", &y.select_rows(&[0, 1, 2, 3, 4, 5, 6]));

            for (i, row) in y.iter_rows().enumerate() {
                let sum_of_squares = row.iter().map(|v| v.powi(2)).sum();
                log_prob[[i, k]] = sum_of_squares;
            }
        }

        log_prob = (log_prob + inputs.cols() as f64 * (2.0 * PI).ln()) * -0.5;
        for (row, det) in log_prob.iter_rows_mut().zip(log_det) {
            for v in row {
                *v += det;
            }
        }
        log_prob
    }

    fn estimate_log_prob_resp(&self, inputs: &Matrix<f64>) -> (Vector<f64>, Matrix<f64>) {
        let mut weighted_log_prob: Matrix<f64> = self.estimate_weighted_log_prob(inputs);
        // length of n_samples
        let log_prob_norm: Vector<f64> = 
            Vector::new(weighted_log_prob.iter_rows().map(|row: &[f64]| {
                let a: f64 = row.iter().map(|v| v.exp()).sum();
                a.ln()
            }).collect::<Vec<f64>>());

        for row in 0..log_prob_norm.size() {
            for col in 0..weighted_log_prob.cols() {
                weighted_log_prob[[row, col]] -= log_prob_norm[row];
            }
        }

        (log_prob_norm, weighted_log_prob)
    }

    fn update_gaussian_parameters(&mut self, inputs: &Matrix<f64>, resp: Matrix<f64>) {
        self.mix_weights = resp.iter_rows()
            .fold(Vector::new(vec![0.0; resp.cols()]), |mut acc, row| {
                for (a, r) in acc.iter_mut().zip(row.iter()) {
                    *a += *r;
                }
                acc
            })
            + 10.0 * EPSILON;
            
        let mut model_means = self.model_means.as_mut().unwrap();
        *model_means = resp.transpose() * inputs;

        for (nk, row) in self.mix_weights.iter().zip(model_means.iter_rows_mut()) {
            for v in row {
                *v /= *nk;
            }
        }

        // Delegate updating the covariance matrices to the cov_option
        let mut model_covars = self.model_covars.as_mut().unwrap();
        self.cov_option.update_covars(model_covars, resp, inputs,
                                      model_means, &self.mix_weights);

        self.mix_weights /= inputs.rows() as f64;
    }
}

/// Contains various methods for calculating covariances
pub trait CovOption<T: Initializer> {
    /// Calculate a precision matrix from a covariance matrix
    fn compute_precision(&self, covariances: &Vec<Matrix<f64>>) -> 
        LearningResult<Vec<Matrix<f64>>>;

    /// Update covariance matrices
    fn update_covars(&self, model_covars: &mut Vec<Matrix<f64>>, resp: Matrix<f64>,
                     inputs: &Matrix<f64>, model_means: &Matrix<f64>,
                     mix_weights: &Vector<f64>);

    /// Regularization constant added along the diagonal
    fn reg_covar(&self) -> f64 {
        0.0
    }
}

/// Performs calculations on a full covariance matrix using Cholesky factorization,
/// adding a constant regularization factor along the diagonal to ensure stability.
#[derive(Clone, Copy, Debug)]
pub struct CholeskyFull {
    /// Regularization constant
    pub reg_covar: f64
}

/// Only calculates the diagonal (the variance). This assumes that all parameters 
/// are uncorrelated. Adds a constant regularization factor (generally unnecessary).
#[derive(Clone, Copy, Debug)]
pub struct Diagonal {
    /// Regularization constant
    pub reg_covar: f64
}

impl CholeskyFull {
    /// Initialize with a covariance regularization constant
    pub fn new(reg_covar: f64) -> Self {
        CholeskyFull { 
            reg_covar: reg_covar 
        }
    }

    /// Solves a system given the Cholesky decomposition of the lower triangular matrix
    pub fn solve_cholesky_decomposition(mut cov_chol: Matrix<f64>) -> Matrix<f64> {
        let half = (cov_chol.rows() as f64 / 2.0).floor() as usize;
        let lower_rows = cov_chol.rows() - half - 1;
        let n_col = cov_chol.cols() - 1;

        let det: f64 = cov_chol.det();
        cov_chol /= det;

        for idx in 0..half {
            // Mirror along the diagonal
            let (mut upper, mut lower) = cov_chol.split_at_mut(half, Axes::Row);
            {
                let swap_row = lower_rows - idx;
                let swap_col = n_col - idx;
                mem::swap(&mut upper[[idx, idx]], &mut lower[[swap_row, swap_col]]);
            }

            // Transpose and invert all other values
            for j in (idx+1)..(n_col+1) {
                let swap_row = lower_rows - idx;
                let swap_col = n_col - j;
                mem::swap(&mut upper[[idx, j]], &mut lower[[swap_row, swap_col]]);
                upper[[idx, j]] *= -1.0;
            }
        }
        cov_chol
    }
}

impl Default for CholeskyFull {
    /// Initialize with a covariance regularization constant of 0.0
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl<T: Initializer> CovOption<T> for CholeskyFull {
    fn compute_precision(&self, covariances: &Vec<Matrix<f64>>) -> LearningResult<Vec<Matrix<f64>>> {
        let n_components = covariances.len();
        let mut precisions_chol = Vec::<Matrix<f64>>::with_capacity(n_components);

        for covariance in covariances {
            let cov_chol: Matrix<f64> = try!(covariance.cholesky());
            precisions_chol.push(Self::solve_cholesky_decomposition(cov_chol));
        }

        Ok(precisions_chol)
    }

    fn update_covars(&self, model_covars: &mut Vec<Matrix<f64>>, resp: Matrix<f64>,
                     inputs: &Matrix<f64>, model_means: &Matrix<f64>,
                     mix_weights: &Vector<f64>) {
        for (k, covariance) in model_covars.iter_mut().enumerate() {
            let mut diff: Matrix<f64> = inputs.clone();
            for (_, mut row) in diff.iter_rows_mut().enumerate() {
                for (i, v) in row.iter_mut().enumerate() {
                    *v -= model_means[[k, i]];
                }
            }

            let mut diff_transpose = diff.transpose();
            let resp_transpose_row = resp.select_cols(&[k]);
            for row in diff_transpose.iter_rows_mut() {
                for (v, x) in row.iter_mut().zip(resp_transpose_row.iter()) {
                    *v *= *x;
                }
            }

            *covariance = (diff_transpose * diff) / mix_weights[k];

            // Add the regularization value
            for idx in 0..covariance.rows() {
                covariance[[idx, idx]] += self.reg_covar;
            }
        }
    }

    fn reg_covar(&self) -> f64 {
        self.reg_covar
    }
}

impl Default for Diagonal {
    /// Initialize with a covariance regularization constant of 0.0
    fn default() -> Self {
        Diagonal { 
            reg_covar: 0.0
        }
    }
}


impl<T: Initializer> CovOption<T> for Diagonal {
    fn compute_precision(&self, covariances: &Vec<Matrix<f64>>) -> LearningResult<Vec<Matrix<f64>>> {
        let n_components = covariances.len();
        let n_features = covariances[0].cols();
        for covariance in covariances {
            for i in 0..covariance.cols() {
                if covariance[[i, i]] < EPSILON {
                    return Err(Error::new(
                            ErrorKind::InvalidState, 
                            "Mixture model had zeros along the diagonals."));
                }

                if covariance[[i, i]].is_nan() {
                    return Err(Error::new(
                            ErrorKind::InvalidState, 
                            "Mixture model had NaN along the diagonals."));
                }
            }
        }

        let mut precisions_chol = Vec::<Matrix<f64>>::with_capacity(n_components);
        for covariance in covariances {
            let v: Vec<f64> = covariance.iter().map(|v| {
                if *v != 0.0 {
                    1.0 / v.sqrt()
                } else {
                    0.0
                }
            }).collect();
            precisions_chol.push(Matrix::<f64>::new(n_features, n_features, v));
        }

        Ok(precisions_chol)
    }

    fn update_covars(&self, model_covars: &mut Vec<Matrix<f64>>, _resp: Matrix<f64>,
                     inputs: &Matrix<f64>, model_means: &Matrix<f64>,
                     mix_weights: &Vector<f64>) {
        for (k, covariance) in model_covars.iter_mut().enumerate() {
            let mut diff: Matrix<f64> = inputs.clone();
            for (_, mut row) in diff.iter_rows_mut().enumerate() {
                for (i, v) in row.iter_mut().enumerate() {
                    *v -= model_means[[k, i]];
                }
            }

            *covariance = Matrix::from_diag(
                &((diff.variance(Axes::Row).unwrap() / mix_weights[k]) 
                 + self.reg_covar).data()[..]);
        }
    }

    fn reg_covar(&self) -> f64 {
        self.reg_covar
    }
}

/// Trait for possible methods of initializing the responsibilities matrix
pub trait Initializer {
    /// Should return the responsibilities matrix: 
    /// shape is [samples, components]
    fn init_resp(k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>>;
}

/// Initialize the responsibilities matrix using randomly generated values
#[derive(Debug)]
pub struct Random;

impl Initializer for Random {
    fn init_resp(k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        use rand::distributions::{IndependentSample, Range};
        let between = Range::new(0.0f64, 1.);
        let mut rng = rand::thread_rng();
        let random_numbers: Vec<f64> = 
            (0..(inputs.rows()*k)).map(|_| between.ind_sample(&mut rng).exp()).collect();
        let mut resp = Matrix::new(inputs.rows(), k, random_numbers);
        let sum = resp.sum_cols();
        for row in resp.iter_rows_mut() {
            for (v, s) in row.iter_mut().zip(sum.iter()) {
                *v /= *s;
            }
        }
        Ok(resp)
    }
}

/// Initialize the responsibilities matrix using k-means clustering
#[derive(Debug)]
pub struct KMeans;

impl Initializer for KMeans {
    fn init_resp(k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        use learning::k_means::KMeansClassifier;
        let mut model = KMeansClassifier::new(k);
        try!(model.train(inputs));
        let classes = try!(model.predict(inputs));
        let mut resp: Matrix<f64> = Matrix::zeros(inputs.rows(), k);
        for (row, col) in classes.iter().enumerate() {
            resp[[row, *col]] = 1.;
        }
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::{GaussianMixtureModel, Random, CholeskyFull};
    use linalg::Vector;

    #[test]
    fn test_means_none() {
        let model = GaussianMixtureModel::<Random, CholeskyFull>::new(5);

        assert_eq!(model.means(), None);
    }

    #[test]
    fn test_covars_none() {
        let model = GaussianMixtureModel::<Random, CholeskyFull>::new(5);

        assert_eq!(model.covariances(), None);
    }

    #[test]
    fn test_negative_mixtures() {
        let mix_weights = Vector::new(vec![-0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::<Random, CholeskyFull>::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }

    #[test]
    fn test_wrong_length_mixtures() {
        let mix_weights = Vector::new(vec![0.1, 0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::<Random, CholeskyFull>::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }
}
