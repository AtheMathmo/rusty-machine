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
//! use rusty_machine::data::transforms::{Transformer, TransformFitter, StandardizerFitter};
//! use rusty_machine::linalg::Matrix;
//!
//! let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 1.5, 3.0]);
//!
//! // Constructs a new `Standardizer` to map to mean 0 and standard
//! // deviation of 1.
//! let mut transformer = StandardizerFitter::default().fit(&inputs).unwrap();
//!
//! // Transform the inputs to get output data with required mean and
//! // standard deviation.
//! let transformed = transformer.transform(inputs).unwrap();
//! ```

use learning::LearningResult;
use learning::error::{Error, ErrorKind};
use linalg::{Matrix, Vector, Axes, BaseMatrix, BaseMatrixMut};
use super::{Invertible, Transformer, TransformFitter};

use rulinalg::utils;

use libnum::{Float, FromPrimitive};

/// A builder used to construct a `Standardizer`
#[derive(Debug)]
pub struct StandardizerFitter<T: Float> {
    scaled_mean: T,
    scaled_stdev: T
}

impl<T: Float> Default for StandardizerFitter<T> {
    fn default() -> Self {
        StandardizerFitter {
            scaled_mean: T::zero(),
            scaled_stdev: T::one()
        }
    }
}

impl<T: Float> StandardizerFitter<T> {
    /// Construct a new `StandardizerFitter` with
    /// specified mean and standard deviation.
    ///
    /// Note that this function does not create a `Transformer`
    /// only a builder which can be used to produce a fitted `Transformer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::data::transforms::StandardizerFitter;
    /// use rusty_machine::linalg::Matrix;
    ///
    /// let fitter = StandardizerFitter::new(0.0, 1.0);
    ///
    /// // We can call `fit` from the `transform::TransformFitter`
    /// // trait to create a `Standardizer` used to actually transform data.
    /// use rusty_machine::data::transforms::TransformFitter;
    /// let mat = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 5.0]);
    /// let transformer = fitter.fit(&mat);
    /// ```
    pub fn new(mean: T, stdev: T) -> StandardizerFitter<T> {
        StandardizerFitter {
            scaled_mean: mean,
            scaled_stdev: stdev
        }
    }
}

impl<T: Float + FromPrimitive> TransformFitter<Matrix<T>, Standardizer<T>> for StandardizerFitter<T> {
    fn fit(self, inputs: &Matrix<T>) -> LearningResult<Standardizer<T>> {
        if inputs.rows() <= 1 {
            Err(Error::new(ErrorKind::InvalidData,
                           "Cannot standardize data with only one row."))
        } else {
            let mean = inputs.mean(Axes::Row);
            let variance = inputs.variance(Axes::Row).map_err(|_| {
                Error::new(ErrorKind::InvalidData, "Cannot compute variance of data.")
            })?;

            if mean.data().iter().any(|x| !x.is_finite()) {
                return Err(Error::new(ErrorKind::InvalidData, "Some data point is non-finite."));
            }

            Ok(Standardizer {
                means: mean,
                variances: variance,
                scaled_mean: self.scaled_mean,
                scaled_stdev: self.scaled_stdev
            })
        }
    }
}

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
    means: Vector<T>,
    /// Variances per column of input data
    variances: Vector<T>,
    /// The mean of the new data (default 0)
    scaled_mean: T,
    /// The standard deviation of the new data (default 1)
    scaled_stdev: T,
}

impl<T: Float + FromPrimitive> Transformer<Matrix<T>> for Standardizer<T> {
    fn transform(&mut self, mut inputs: Matrix<T>) -> LearningResult<Matrix<T>> {
        if self.means.size() != inputs.cols() {
            Err(Error::new(ErrorKind::InvalidData,
                            "Input data has different number of columns from fitted data."))
        } else {
            for mut row in inputs.row_iter_mut() {
                // Subtract the mean
                utils::in_place_vec_bin_op(row.raw_slice_mut(), self.means.data(), |x, &y| *x = *x - y);
                utils::in_place_vec_bin_op(row.raw_slice_mut(), self.variances.data(), |x, &y| {
                    *x = (*x * self.scaled_stdev / y.sqrt()) + self.scaled_mean
                });
            }
            Ok(inputs)
        }
    }
}

impl<T: Float + FromPrimitive> Invertible<Matrix<T>> for Standardizer<T> {
    fn inv_transform(&self, mut inputs: Matrix<T>) -> LearningResult<Matrix<T>> {
        let features = self.means.size();
        if inputs.cols() != features {
            return Err(Error::new(ErrorKind::InvalidData,
                                    "Inputs have different feature count than transformer."));
        }

        for mut row in inputs.row_iter_mut() {
            utils::in_place_vec_bin_op(row.raw_slice_mut(), self.variances.data(), |x, &y| {
                *x = (*x - self.scaled_mean) * y.sqrt() / self.scaled_stdev
            });

            // Add the mean
            utils::in_place_vec_bin_op(row.raw_slice_mut(), self.means.data(), |x, &y| *x = *x + y);
        }

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Transformer, TransformFitter, Invertible};
    use linalg::{Axes, Matrix};

    use std::f64;

    #[test]
    fn single_row_test() {
        let inputs = Matrix::new(1, 2, vec![1.0, 2.0]);

        let standardizer = StandardizerFitter::default();
        let transformer = standardizer.fit(&inputs);
        assert!(transformer.is_err());
    }

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::NAN; 4]);

        let standardizer = StandardizerFitter::default();
        let transformer = standardizer.fit(&inputs);
        assert!(transformer.is_err());
    }

    #[test]
    fn inf_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::INFINITY; 4]);

        let standardizer = StandardizerFitter::default();
        let transformer = standardizer.fit(&inputs);
        assert!(transformer.is_err());
    }

    #[test]
    fn wrong_transform_size_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = StandardizerFitter::default().fit(&inputs).unwrap();
        let res = standardizer.transform(matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0]);
        assert!(res.is_err());
    }

    #[test]
    fn basic_standardize_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = StandardizerFitter::default().fit(&inputs).unwrap();
        let transformed = standardizer.transform(inputs).unwrap();

        let new_mean = transformed.mean(Axes::Row);
        let new_var = transformed.variance(Axes::Row).unwrap();

        assert!(new_mean.data().iter().all(|x| x.abs() < 1e-5));
        assert!(new_var.data().iter().all(|x| (x.abs() - 1.0) < 1e-5));
    }

    #[test]
    fn custom_standardize_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = StandardizerFitter::new(1.0, 2.0).fit(&inputs).unwrap();
        let transformed = standardizer.transform(inputs).unwrap();

        let new_mean = transformed.mean(Axes::Row);
        let new_var = transformed.variance(Axes::Row).unwrap();

        assert!(new_mean.data().iter().all(|x| (x.abs() - 1.0) < 1e-5));
        assert!(new_var.data().iter().all(|x| (x.abs() - 4.0) < 1e-5));
    }

    #[test]
    fn inv_transform_identity_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut standardizer = StandardizerFitter::new(1.0, 3.0).fit(&inputs).unwrap();
        let transformed = standardizer.transform(inputs.clone()).unwrap();

        let original = standardizer.inv_transform(transformed).unwrap();

        assert!((inputs - original).data().iter().all(|x| x.abs() < 1e-5));
    }
}
