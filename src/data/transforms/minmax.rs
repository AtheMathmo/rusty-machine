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
//! use rusty_machine::data::transforms::{Transformer, MinMaxScaler};
//! use rusty_machine::linalg::Matrix;
//!
//! // Constructs a new `MinMaxScaler` to map minimum to 0 and maximum
//! // to 1.
//! let mut transformer = MinMaxScaler::default();
//!
//! let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 1.5, 3.0]);
//!
//! // Transform the inputs to get output data with required minimum
//! // and maximum.
//! let transformed = transformer.transform(inputs).unwrap();
//! ```

use learning::error::{Error, ErrorKind};
use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use super::{Invertible, Transformer};

use rulinalg::utils;

use libnum::Float;

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
    scale_factors: Option<Vec<T>>,
    /// Values to add to each column after scaling
    const_factors: Option<Vec<T>>,
    /// The min of the new data (default 0)
    scaled_min: T,
    /// The max of the new data (default 1)
    scaled_max: T,
}

/// Create a default `MinMaxScaler` with minimum of `0` and
/// maximum of `1`.
impl<T: Float> Default for MinMaxScaler<T> {
    fn default() -> MinMaxScaler<T> {
        MinMaxScaler::new(T::zero(), T::one())
    }
}

impl<T: Float> MinMaxScaler<T> {
    /// Constructs a new MinMaxScaler with the specified scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::data::transforms::{MinMaxScaler, Transformer};
    ///
    /// // Constructs a new `MinMaxScaler` which will give the data
    /// // minimum `0` and maximum `2`.
    /// let transformer = MinMaxScaler::new(0.0, 2.0);
    /// ```
    pub fn new(min: T, max: T) -> MinMaxScaler<T> {
        MinMaxScaler {
            scale_factors: None,
            const_factors: None,
            scaled_min: min,
            scaled_max: max,
        }
    }
}

impl<T: Float> Transformer<Matrix<T>> for MinMaxScaler<T> {
    fn transform(&mut self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        let features = inputs.cols();

        let mut input_min_max = vec![(T::max_value(), T::min_value()); features];

        for row in inputs.iter_rows() {
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
        let scales = try!(input_min_max.iter()
            .map(|&(x, y)| {
                let s = (self.scaled_max - self.scaled_min) / (y - x);
                if s.is_finite() {
                    Ok(s)
                } else {
                    Err(Error::new(ErrorKind::InvalidData,
                                   "Constant feature columns not supported."))
                }
            })
            .collect::<Result<Vec<_>, _>>());

        let consts = input_min_max.iter()
            .zip(scales.iter())
            .map(|(&(_, x), &s)| self.scaled_max - x * s)
            .collect::<Vec<_>>();

        for row in inputs.iter_rows_mut() {
            utils::in_place_vec_bin_op(row, &scales, |x, &y| {
                *x = *x * y;
            });

            utils::in_place_vec_bin_op(row, &consts, |x, &y| {
                *x = *x + y;
            });
        }

        self.scale_factors = Some(scales);
        self.const_factors = Some(consts);

        Ok(inputs)
    }
}

impl<T: Float> Invertible<Matrix<T>> for MinMaxScaler<T> {
    fn inv_transform(&self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        if let (&Some(ref scales), &Some(ref consts)) = (&self.scale_factors, &self.const_factors) {

            let features = scales.len();
            if inputs.cols() != features {
                return Err(Error::new(ErrorKind::InvalidData,
                                      "Inputs have different feature count than transformer."));
            }

            for row in inputs.iter_rows_mut() {
                for i in 0..features {
                    row[i] = (row[i] - consts[i]) / scales[i];
                }
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
    use super::super::{Transformer, Invertible};
    use linalg::Matrix;
    use std::f64;

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::NAN; 4]);

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        let res = scaler.transform(inputs);

        assert!(res.is_err());
    }

    #[test]
    fn infinity_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::INFINITY; 4]);

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        let res = scaler.transform(inputs);

        assert!(res.is_err());
    }

    #[test]
    fn basic_scale_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
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

        let mut scaler = MinMaxScaler::new(1.0, 3.0);
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

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        let res = scaler.transform(inputs);

        assert!(res.is_err());
    }

    #[test]
    fn inv_transform_identity_test() {
        let inputs = Matrix::new(2, 2, vec![-1.0f32, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxScaler::new(1.0, 3.0);
        let transformed = scaler.transform(inputs.clone()).unwrap();

        let original = scaler.inv_transform(transformed).unwrap();

        assert!((inputs - original).data().iter().all(|x| x.abs() < 1e-5));
    }
}
