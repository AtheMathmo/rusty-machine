//! Naive Bayes Classifiers
//!
//! The classifier supports Gaussian, Bernoulli and Multinomial distributions.
//!
//! A naive Bayes classifier works by treating the features of each input as independent
//! observations. Under this assumption we utilize Bayes' rule to compute the
//! probability that each input belongs to a given class.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::naive_bayes::{NaiveBayes, Gaussian};
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::learning::SupModel;
//!
//! let inputs = Matrix::new(6, 2, vec![1.0, 1.1,
//!                                     1.1, 0.9,
//!                                     2.2, 2.3,
//!                                     2.5, 2.7,
//!                                     5.2, 4.3,
//!                                     6.2, 7.3]);
//!
//! let targets = Matrix::new(6,3, vec![1.0, 0.0, 0.0,
//!                                     1.0, 0.0, 0.0,
//!                                     0.0, 1.0, 0.0,
//!                                     0.0, 1.0, 0.0,
//!                                     0.0, 0.0, 1.0,
//!                                     0.0, 0.0, 1.0]);
//!
//! // Create a Gaussian Naive Bayes classifier.
//! let mut model = NaiveBayes::<Gaussian>::new();
//!
//! // Train the model.
//! model.train(&inputs, &targets).unwrap();
//!
//! // Predict the classes on the input data
//! let outputs = model.predict(&inputs).unwrap();
//!
//! // Will output the target classes - otherwise our classifier is bad!
//! println!("Final outputs --\n{}", outputs);
//! ```

use linalg::{Matrix, Axes, BaseMatrix, BaseMatrixMut};
use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};
use rulinalg::utils;

use std::f64::consts::PI;

/// The Naive Bayes model.
#[derive(Debug, Default)]
pub struct NaiveBayes<T: Distribution> {
    distr: Option<T>,
    cluster_count: Option<usize>,
    class_prior: Option<Vec<f64>>,
    class_counts: Vec<usize>,
}

impl<T: Distribution> NaiveBayes<T> {
    /// Create a new NaiveBayes model from a given
    /// distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::naive_bayes::{NaiveBayes, Gaussian};
    ///
    /// // Create a new Gaussian Naive Bayes model.
    /// let _ = NaiveBayes::<Gaussian>::new();
    /// ```
    pub fn new() -> NaiveBayes<T> {
        NaiveBayes {
            distr: None,
            cluster_count: None,
            class_prior: None,
            class_counts: Vec::new(),
        }
    }

    /// Get the cluster count for this model.
    ///
    /// Returns an option which is `None` until the model has been trained.
    pub fn cluster_count(&self) -> Option<&usize> {
        self.cluster_count.as_ref()
    }

    /// Get the class prior distribution for this model.
    ///
    /// Returns an option which is `None` until the model has been trained.
    pub fn class_prior(&self) -> Option<&Vec<f64>> {
        self.class_prior.as_ref()
    }

    /// Get the distribution for this model.
    ///
    /// Returns an option which is `None` until the model has been trained.
    pub fn distr(&self) -> Option<&T> {
        self.distr.as_ref()
    }
}

/// Train and predict from the Naive Bayes model.
///
/// The input matrix must be rows made up of features.
/// The target matrix should have indicator vectors in each row specifying
/// the input class. e.g. [[1,0,0],[0,0,1]] shows class 1 first, then class 3.
impl<T: Distribution> SupModel<Matrix<f64>, Matrix<f64>> for NaiveBayes<T> {
    /// Train the model using inputs and targets.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) -> LearningResult<()> {
        self.distr = Some(T::from_model_params(targets.cols(), inputs.cols()));
        self.update_params(inputs, targets)
    }

    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        let log_probs = self.get_log_probs(inputs)?;
        let input_classes = NaiveBayes::<T>::get_classes(log_probs);

        if let Some(cluster_count) = self.cluster_count {
            let mut class_data = Vec::with_capacity(inputs.rows() * cluster_count);

            for c in input_classes {
                let mut row = vec![0f64; cluster_count];
                row[c] = 1f64;

                class_data.append(&mut row);
            }

            Ok(Matrix::new(inputs.rows(), cluster_count, class_data))
        } else {
            Err(Error::new(ErrorKind::UntrainedModel, "The model has not been trained."))
        }
    }
}

impl<T: Distribution> NaiveBayes<T> {
    /// Get the log-probabilities per class for each input.
    pub fn get_log_probs(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {

        if let (&Some(ref distr), &Some(ref prior)) = (&self.distr, &self.class_prior) {
            // Get the joint log likelihood from the distribution
            distr.joint_log_lik(inputs, prior)
        } else {
            Err(Error::new_untrained())
        }
    }

    fn update_params(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) -> LearningResult<()> {
        let class_count = targets.cols();
        let total_data = inputs.rows();

        self.class_counts = vec![0; class_count];
        let mut class_data = vec![Vec::new(); class_count];

        for (idx, row) in targets.row_iter().enumerate() {
            // Find the class of this input
            let class = NaiveBayes::<T>::find_class(row.raw_slice())?;

            // Note the class of the input
            class_data[class].push(idx);
            self.class_counts[class] += 1;
        }

        if let Some(ref mut distr) = self.distr {
            for (idx, c) in class_data.into_iter().enumerate() {
                // If this class' vector has not been populated, we can safely
                // skip this iteration, since the user is clearly not interested
                // in associating features with this class
                if c.is_empty() {
                    continue;
                }
                // Update the parameters within this class
                distr.update_params(&inputs.select_rows(&c), idx)?;
            }
        }

        let mut class_prior = Vec::with_capacity(class_count);

        // Compute the prior as the proportion in each class
        class_prior.extend(self.class_counts.iter().map(|c| *c as f64 / total_data as f64));

        self.class_prior = Some(class_prior);
        self.cluster_count = Some(class_count);
        Ok(())
    }

    fn find_class(row: &[f64]) -> LearningResult<usize> {
        // Find the `1` entry in the row
        for (idx, r) in row.into_iter().enumerate() {
            if *r == 1f64 {
                return Ok(idx);
            }
        }

        Err(Error::new(ErrorKind::InvalidState,
                       "No class found for entry in targets"))
    }

    fn get_classes(log_probs: Matrix<f64>) -> Vec<usize> {
        let mut data_classes = Vec::with_capacity(log_probs.rows());

        data_classes.extend(log_probs.row_iter().map(|row| {
            // Argmax each class log-probability per input
            let (class, _) = utils::argmax(row.raw_slice());
            class
        }));

        data_classes
    }
}

/// Naive Bayes Distribution.
pub trait Distribution {
    /// Initialize the distribution parameters.
    fn from_model_params(class_count: usize, features: usize) -> Self;

    /// Updates the distribution parameters.
    fn update_params(&mut self, data: &Matrix<f64>, class: usize) -> LearningResult<()>;

    /// Compute the joint log likelihood of the data.
    ///
    /// Returns a matrix with rows containing the probability that the input lies in each class.
    fn joint_log_lik(&self,
                     data: &Matrix<f64>,
                     class_prior: &[f64])
                     -> LearningResult<Matrix<f64>>;
}

/// The Gaussian Naive Bayes model distribution.
///
/// Defines:
///
/// p(x|C<sub>k</sub>) = ∏<sub>i</sub> N(x<sub>i</sub> ;
/// μ<sub>k</sub>, σ<sup>2</sup><sub>k</sub>)
#[derive(Debug)]
pub struct Gaussian {
    theta: Matrix<f64>,
    sigma: Matrix<f64>,
}

impl Gaussian {
    /// Returns the distribution means.
    ///
    /// This is a matrix of class by feature means.
    pub fn theta(&self) -> &Matrix<f64> {
        &self.theta
    }

    /// Returns the distribution variances.
    ///
    /// This is a matrix of class by feature variances.
    pub fn sigma(&self) -> &Matrix<f64> {
        &self.sigma
    }
}

impl Distribution for Gaussian {
    fn from_model_params(class_count: usize, features: usize) -> Gaussian {
        Gaussian {
            theta: Matrix::zeros(class_count, features),
            sigma: Matrix::zeros(class_count, features),
        }
    }

    fn update_params(&mut self, data: &Matrix<f64>, class: usize) -> LearningResult<()> {
        // Compute mean and sample variance
        let mean = data.mean(Axes::Row).into_vec();
        let var = data.variance(Axes::Row).map_err(|_| {
                Error::new(ErrorKind::InvalidData,
                           "Cannot compute variance for Gaussian distribution.")
            })?
            .into_vec();

        let features = data.cols();

        for (idx, (m, v)) in mean.into_iter().zip(var.into_iter()).enumerate() {
            self.theta.mut_data()[class * features + idx] = m;
            self.sigma.mut_data()[class * features + idx] = v;
        }

        Ok(())
    }

    fn joint_log_lik(&self,
                     data: &Matrix<f64>,
                     class_prior: &[f64])
                     -> LearningResult<Matrix<f64>> {
        let class_count = class_prior.len();
        let mut log_lik = Vec::with_capacity(class_count);

        for (i, item) in class_prior.into_iter().enumerate() {
            let joint_i = item.ln();
            let n_ij = -0.5 * (self.sigma.select_rows(&[i]) * 2.0 * PI).apply(&|x| x.ln()).sum();

            // NOTE: Here we are copying the row data which is inefficient
            let r_ij = (data - self.theta.select_rows(&vec![i; data.rows()]))
                .apply(&|x| x * x)
                .elediv(&self.sigma.select_rows(&vec![i; data.rows()]))
                .sum_cols();

            let res = (-r_ij * 0.5) + n_ij;

            log_lik.append(&mut (res + joint_i).into_vec());
        }

        Ok(Matrix::new(class_count, data.rows(), log_lik).transpose())
    }
}

/// The Bernoulli Naive Bayes model distribution.
///
/// Defines:
///
///    p(x|C<sub>k</sub>) = ∏<sub>i</sub> p<sub>k</sub><sup>x<sub>i</sub></sup>
/// (1-p)<sub>k</sub><sup>1-x<sub>i</sub></sup>
#[derive(Debug)]
pub struct Bernoulli {
    log_probs: Matrix<f64>,
    pseudo_count: f64,
}

impl Bernoulli {
    /// The log probability matrix.
    ///
    /// A matrix of class by feature model log-probabilities.
    pub fn log_probs(&self) -> &Matrix<f64> {
        &self.log_probs
    }
}

impl Distribution for Bernoulli {
    fn from_model_params(class_count: usize, features: usize) -> Bernoulli {
        Bernoulli {
            log_probs: Matrix::zeros(class_count, features),
            pseudo_count: 1f64,
        }
    }

    fn update_params(&mut self, data: &Matrix<f64>, class: usize) -> LearningResult<()> {
        let features = data.cols();

        // We add the pseudo count to the class count and feature count
        let pseudo_cc = data.rows() as f64 + (2f64 * self.pseudo_count);
        let pseudo_fc = data.sum_rows() + self.pseudo_count;

        let log_probs = (pseudo_fc.apply(&|x| x.ln()) - pseudo_cc.ln()).into_vec();

        for (i, item) in log_probs.iter().enumerate().take(features) {
            self.log_probs[[class, i]] = *item;
        }

        Ok(())

    }

    fn joint_log_lik(&self,
                     data: &Matrix<f64>,
                     class_prior: &[f64])
                     -> LearningResult<Matrix<f64>> {
        let class_count = class_prior.len();

        let neg_prob = self.log_probs.clone().apply(&|x| (1f64 - x.exp()).ln());

        let res = data * (&self.log_probs - &neg_prob).transpose();

        // NOTE: Some messy stuff now to get the class row contribution.
        // Really we want to add to each row the class log-priors and the
        // neg_prob_sum contribution - the last term in
        // x log(p) + (1-x)log(1-p) = x (log(p) - log(1-p)) + log(1-p)

        let mut per_class_row = Vec::with_capacity(class_count);
        let neg_prob_sum = neg_prob.sum_cols();

        for (idx, p) in class_prior.into_iter().enumerate() {
            per_class_row.push(p.ln() + neg_prob_sum[idx]);
        }

        let class_row_mat = Matrix::new(1, class_count, per_class_row);

        Ok(res + class_row_mat.select_rows(&vec![0; data.rows()]))
    }
}

/// The Multinomial Naive Bayes model distribution.
///
/// Defines:
///
///    p(x|C<sub>k</sub>) ∝ ∏<sub>i</sub> p<sub>k</sub><sup>x<sub>i</sub></sup>
#[derive(Debug)]
pub struct Multinomial {
    log_probs: Matrix<f64>,
    pseudo_count: f64,
}

impl Multinomial {
    /// The log probability matrix.
    ///
    /// A matrix of class by feature model log-probabilities.
    pub fn log_probs(&self) -> &Matrix<f64> {
        &self.log_probs
    }
}

impl Distribution for Multinomial {
    fn from_model_params(class_count: usize, features: usize) -> Multinomial {
        Multinomial {
            log_probs: Matrix::zeros(class_count, features),
            pseudo_count: 1f64,
        }
    }

    fn update_params(&mut self, data: &Matrix<f64>, class: usize) -> LearningResult<()> {
        let features = data.cols();

        let pseudo_fc = data.sum_rows() + self.pseudo_count;
        let pseudo_cc = pseudo_fc.sum();

        let log_probs = (pseudo_fc.apply(&|x| x.ln()) - pseudo_cc.ln()).into_vec();

        for (i, item) in log_probs.iter().enumerate().take(features) {
            self.log_probs[[class, i]] = *item;
        }

        Ok(())
    }

    fn joint_log_lik(&self,
                     data: &Matrix<f64>,
                     class_prior: &[f64])
                     -> LearningResult<Matrix<f64>> {
        let class_count = class_prior.len();

        let res = data * self.log_probs.transpose();

        let mut per_class_row = Vec::with_capacity(class_count);
        for p in class_prior {
            per_class_row.push(p.ln());
        }

        let class_row_mat = Matrix::new(1, class_count, per_class_row);

        Ok(res + class_row_mat.select_rows(&vec![0; data.rows()]))
    }
}

#[cfg(test)]
mod tests {
    use super::NaiveBayes;
    use super::Gaussian;
    use super::Bernoulli;
    use super::Multinomial;

    use learning::SupModel;

    use linalg::Matrix;

    #[test]
    fn test_gaussian() {
        let inputs = Matrix::new(6,
                                 2,
                                 vec![1.0, 1.1, 1.1, 0.9, 2.2, 2.3, 2.5, 2.7, 5.2, 4.3, 6.2, 7.3]);

        let targets = Matrix::new(6,
                                  3,
                                  vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        let mut model = NaiveBayes::<Gaussian>::new();
        model.train(&inputs, &targets).unwrap();

        let outputs = model.predict(&inputs).unwrap();
        assert_eq!(outputs.into_vec(), targets.into_vec());
    }

    #[test]
    fn test_bernoulli() {
        let inputs = Matrix::new(4,
                                 3,
                                 vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

        let targets = Matrix::new(4, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

        let mut model = NaiveBayes::<Bernoulli>::new();
        model.train(&inputs, &targets).unwrap();

        let outputs = model.predict(&inputs).unwrap();
        assert_eq!(outputs.into_vec(), targets.into_vec());
    }

    #[test]
    fn test_multinomial() {
        let inputs = Matrix::new(4,
                                 3,
                                 vec![1.0, 0.0, 5.0, 0.0, 0.0, 11.0, 13.0, 1.0, 0.0, 12.0, 3.0,
                                      0.0]);

        let targets = Matrix::new(4, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

        let mut model = NaiveBayes::<Multinomial>::new();
        model.train(&inputs, &targets).unwrap();

        let outputs = model.predict(&inputs).unwrap();
        assert_eq!(outputs.into_vec(), targets.into_vec());
    }
}
