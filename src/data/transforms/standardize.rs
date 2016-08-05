//! The Standardizing Transformer
//!
//! This module contains the `Standardizer` transformer.
//!
//! The `Standardizer` transformer is used to transform input data
//! so that the mean and standard deviation of each column are as
//! specified. This is commonly used to transform the data to have `0` mean
//! and a standard deviation of `1`.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::transforms::{Transformer, Standardizer};
//! use rusty_machine::linalg::Matrix;
//!
//! // Constructs a new `Standardizer` to map to mean 0 and standard
//! // deviation of 1.
//! let mut transformer = Standardizer::default();
//!
//! let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 1.5, 3.0]);
//!
//! // Transform the inputs to get output data with required mean and
//! // standard deviation.
//! let transformed = transformer.transform(inputs).unwrap();
//! ```

use learning::error::{Error, ErrorKind};
use linalg::{Matrix, Vector, Axes};
use super::Transformer;

use rulinalg::utils;

use libnum::{Float, FromPrimitive};

/// The Standardizer
///
/// The Standardizer provides an implementation of `Transformer`
/// which allows us to transform the input data to have a new mean
/// and standard deviation.
///
/// See the module description for more information.
#[derive(Debug)]
pub struct Standardizer<T: Float> {
    /// Means per column of input data
    means: Option<Vector<T>>,
    /// Variances per column of input data
    variances: Option<Vector<T>>,
    /// The mean of the new data (default 0)
    scaled_mean: T,
    /// The standard deviation of the new data (default 1)
    scaled_stdev: T,
}

/// Create a `Standardizer` with mean `0` and standard
/// deviation `1`.
impl<T: Float> Default for Standardizer<T> {
    fn default() -> Standardizer<T> {
        Standardizer {
            means: None,
            variances: None,
            scaled_mean: T::zero(),
            scaled_stdev: T::one(),
        }
    }
}

impl<T: Float> Standardizer<T> {
    /// Constructs a new `Standardizer` with the given mean and variance
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::data::transforms::Standardizer;
    ///
    /// // Constructs a new `Standardizer` which will give the data
    /// // mean `0` and standard deviation `2`.
    /// let transformer = Standardizer::new(0.0, 2.0);
    /// ```
    pub fn new(mean: T, stdev: T) -> Standardizer<T> {
        Standardizer {
            means: None,
            variances: None,
            scaled_mean: mean,
            scaled_stdev: stdev,
        }
    }
}

impl<T: Float + FromPrimitive> Transformer<Matrix<T>> for Standardizer<T> {
    fn transform(&mut self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        if inputs.rows() <= 1 {
            Err(Error::new(ErrorKind::InvalidData,
                           "Cannot standardize data with only one row."))
        } else {
            let mean = inputs.mean(Axes::Row);
            let variance = inputs.variance(Axes::Row);

            if mean.data().iter().any(|x| !x.is_finite()) {
                return Err(Error::new(ErrorKind::InvalidData, "Some data point is non-finite."));
            }

            for row in inputs.iter_rows_mut() {
                // Subtract the mean
                utils::in_place_vec_bin_op(row, &mean.data(), |x, &y| *x = *x - y);
                utils::in_place_vec_bin_op(row, &variance.data(), |x, &y| {
                    *x = (*x * self.scaled_stdev / y.sqrt()) + self.scaled_mean
                });
            }

            self.means = Some(mean);
            self.variances = Some(variance);
            Ok(inputs)
        }
    }

    fn inv_transform(&self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        if let (&Some(ref means), &Some(ref variances)) = (&self.means, &self.variances) {

            let features = means.size();
            if inputs.cols() != features {
                return Err(Error::new(ErrorKind::InvalidData,
                                      "Inputs have different feature count than transformer."));
            }

            for row in inputs.iter_rows_mut() {
                utils::in_place_vec_bin_op(row, &variances.data(), |x, &y| {
                    *x = (*x - self.scaled_mean) * y.sqrt() / self.scaled_stdev
                });

                // Add the mean
                utils::in_place_vec_bin_op(row, &means.data(), |x, &y| *x = *x + y);
            }

            Ok(inputs)
        } else {
            Err(Error::new(ErrorKind::InvalidState, "Transformer has not been fitted."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Transformer;
    use linalg::{Axes, Matrix};

    use std::f64;

    #[test]
    fn single_row_test() {
        let inputs = Matrix::new(1, 2, vec![1.0, 2.0]);

        let mut standardizer = Standardizer::default();

        let res = standardizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::NAN; 4]);

        let mut standardizer = Standardizer::default();

        let res = standardizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn inf_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::INFINITY; 4]);

        let mut standardizer = Standardizer::default();

        let res = standardizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn basic_standardize_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = Standardizer::default();
        let transformed = standardizer.transform(inputs).unwrap();

        let new_mean = transformed.mean(Axes::Row);
        let new_var = transformed.variance(Axes::Row);

        assert!(new_mean.data().iter().all(|x| x.abs() < 1e-5));
        assert!(new_var.data().iter().all(|x| (x.abs() - 1.0) < 1e-5));
    }

    #[test]
    fn custom_standardize_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = Standardizer::new(1.0, 2.0);
        let transformed = standardizer.transform(inputs).unwrap();

        let new_mean = transformed.mean(Axes::Row);
        let new_var = transformed.variance(Axes::Row);

        assert!(new_mean.data().iter().all(|x| (x.abs() - 1.0) < 1e-5));
        assert!(new_var.data().iter().all(|x| (x.abs() - 4.0) < 1e-5));
    }

    #[test]
    fn inv_transform_identity_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = Standardizer::new(1.0, 3.0);
        let transformed = standardizer.transform(inputs.clone()).unwrap();

        let original = standardizer.inv_transform(transformed).unwrap();

        assert!((inputs - original).data().iter().all(|x| x.abs() < 1e-5));
    }
}