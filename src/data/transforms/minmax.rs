//! The Min-Max transformer
//!
//! This module contains the `MinMaxScaler` transformer.
//!
//! The `MinMaxScaler` transformer is used to transform input data
//! so that the minimum and maximum of each column are as specified.
//! This is commonly used to transform the data to have a minimum of
//! `0` and a maximum of `1`.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::transforms::{Transformer, TransformFitter, MinMaxFitter};
//! use rusty_machine::linalg::Matrix;
//!
//! let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 1.5, 3.0]);
//!
//! // Constructs a new `MinMaxScaler` to map minimum to 0 and maximum
//! // to 1.
//! let mut transformer = MinMaxFitter::default().fit(&inputs).unwrap();
//!
//!
//! // Transform the inputs to get output data with required minimum
//! // and maximum.
//! let transformed = transformer.transform(inputs).unwrap();
//! ```

use learning::error::{Error, ErrorKind};
use learning::LearningResult;
use linalg::{Matrix, BaseMatrix, BaseMatrixMut, Vector};
use super::{Invertible, Transformer, TransformFitter};

use rulinalg::utils;

use libnum::Float;

/// A builder used to construct a `MinMaxScaler`
#[derive(Debug)]
pub struct MinMaxFitter<T: Float> {
    scaled_min: T,
    scaled_max: T
}

impl<T: Float> Default for MinMaxFitter<T> {
    fn default() -> Self {
        MinMaxFitter {
            scaled_min: T::zero(),
            scaled_max: T::one()
        }
    }
}

impl<T: Float> MinMaxFitter<T> {
    /// Construct a new `MinMaxFitter` with
    /// specified mean and standard deviation.
    ///
    /// Note that this function does not create a `Transformer`
    /// only a builder which can be used to produce a fitted `Transformer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::data::transforms::MinMaxFitter;
    /// use rusty_machine::linalg::Matrix;
    ///
    /// let fitter = MinMaxFitter::new(0.0, 1.0);
    ///
    /// // We can call `fit` from the `transform::TransformFitter`
    /// // trait to create a `MinMaxScaler` used to actually transform data.
    /// use rusty_machine::data::transforms::TransformFitter;
    /// let mat = Matrix::new(2,2,vec![1.0, 2.0, 3.0, 5.0]);
    /// let transformer = fitter.fit(&mat);
    /// ```
    pub fn new(min: T, max: T) -> Self {
        MinMaxFitter {
            scaled_min: min,
            scaled_max: max
        }
    }
}

impl<T: Float> TransformFitter<Matrix<T>, MinMaxScaler<T>> for MinMaxFitter<T> {
    fn fit(self, inputs: &Matrix<T>) -> LearningResult<MinMaxScaler<T>> {
        let features = inputs.cols();

        // TODO: can use min, max
        // https://github.com/AtheMathmo/rulinalg/pull/115
        let mut input_min_max = vec![(T::max_value(), T::min_value()); features];

        for row in inputs.row_iter() {
            for (idx, (feature, min_max)) in row.into_iter().zip(input_min_max.iter_mut()).enumerate() {
                if !feature.is_finite() {
                    return Err(Error::new(ErrorKind::InvalidData,
                                          format!("Data point in column {} cannot be \
                                                   processed",
                                                  idx)));
                }
                // Update min
                if *feature < min_max.0 {
                    min_max.0 = *feature;
                }
                // Update max
                if *feature > min_max.1 {
                    min_max.1 = *feature;
                }
            }
        }

        // We'll scale each feature by a * x + b.
        // Where scales holds `a` per column and consts
        // holds `b`.
        let scales = input_min_max.iter()
            .map(|&(x, y)| {
                let s = (self.scaled_max - self.scaled_min) / (y - x);
                if s.is_finite() {
                    Ok(s)
                } else {
                    Err(Error::new(ErrorKind::InvalidData,
                                   "Constant feature columns not supported."))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let consts = input_min_max.iter()
            .zip(scales.iter())
            .map(|(&(_, x), &s)| self.scaled_max - x * s)
            .collect::<Vec<_>>();
        
        Ok(MinMaxScaler {
            scale_factors: Vector::new(scales),
            const_factors: Vector::new(consts)
        })
    }
}

/// The `MinMaxScaler`
///
/// The `MinMaxScaler` provides an implementation of `Transformer`
/// which allows us to transform the input data to have a new minimum
/// and maximum per column.
///
/// See the module description for more information.
#[derive(Debug)]
pub struct MinMaxScaler<T: Float> {
    /// Values to scale each column by
    scale_factors: Vector<T>,
    /// Values to add to each column after scaling
    const_factors: Vector<T>,
}


impl<T: Float> Transformer<Matrix<T>> for MinMaxScaler<T> {
    fn transform(&mut self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        if self.scale_factors.size() != inputs.cols() {
            Err(Error::new(ErrorKind::InvalidData,
                            "Input data has different number of columns than fitted data."))
        } else {
            for mut row in inputs.row_iter_mut() {
                utils::in_place_vec_bin_op(row.raw_slice_mut(), self.scale_factors.data(), |x, &y| {
                    *x = *x * y;
                });

                utils::in_place_vec_bin_op(row.raw_slice_mut(), self.const_factors.data(), |x, &y| {
                    *x = *x + y;
                });
            }
            Ok(inputs)
        }
    }
}

impl<T: Float> Invertible<Matrix<T>> for MinMaxScaler<T> {

    fn inv_transform(&self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        let features = self.scale_factors.size();
        if inputs.cols() != features {
            return Err(Error::new(ErrorKind::InvalidData,
                                    "Input data has different number of columns than fitted data."));
        }

        for mut row in inputs.row_iter_mut() {
            for i in 0..features {
                row[i] = (row[i] - self.const_factors[i]) / self.scale_factors[i];
            }
        }

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Transformer, TransformFitter, Invertible};
    use linalg::Matrix;
    use std::f64;

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::NAN; 4]);

        let res = MinMaxFitter::new(0.0, 1.0).fit(&inputs);
        assert!(res.is_err());
    }

    #[test]
    fn infinity_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::INFINITY; 4]);

        let res = MinMaxFitter::new(0.0, 1.0).fit(&inputs);
        assert!(res.is_err());
    }

    #[test]
    fn basic_scale_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxFitter::new(0.0, 1.0).fit(&inputs).unwrap();
        let transformed = scaler.transform(inputs).unwrap();

        assert!(transformed.data().iter().all(|&x| x >= 0.0));
        assert!(transformed.data().iter().all(|&x| x <= 1.0));

        // First row scales to 0 and second to 1
        transformed[[0, 0]].abs() < 1e-10;
        transformed[[0, 1]].abs() < 1e-10;
        (transformed[[1, 0]] - 1.0).abs() < 1e-10;
        (transformed[[1, 1]] - 1.0).abs() < 1e-10;
    }

    #[test]
    fn custom_scale_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxFitter::new(1.0, 3.0).fit(&inputs).unwrap();
        let transformed = scaler.transform(inputs).unwrap();

        assert!(transformed.data().iter().all(|&x| x >= 1.0));
        assert!(transformed.data().iter().all(|&x| x <= 3.0));

        // First row scales to 1 and second to 3
        (transformed[[0, 0]] - 1.0).abs() < 1e-10;
        (transformed[[0, 1]] - 1.0).abs() < 1e-10;
        (transformed[[1, 0]] - 3.0).abs() < 1e-10;
        (transformed[[1, 1]] - 3.0).abs() < 1e-10;
    }

    #[test]
    fn constant_feature_test() {
        let inputs = Matrix::new(2, 2, vec![1.0, 2.0, 1.0, 3.0]);

        let res = MinMaxFitter::new(0.0, 1.0).fit(&inputs);
        assert!(res.is_err());
    }

    #[test]
    fn inv_transform_identity_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxFitter::new(1.0, 3.0).fit(&inputs).unwrap();
        let transformed = scaler.transform(inputs.clone()).unwrap();

        let original = scaler.inv_transform(transformed).unwrap();

        assert!((inputs - original).data().iter().all(|x| x.abs() < 1e-5));
    }
}
