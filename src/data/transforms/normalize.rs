//! The Normalizing Transformer
//!
//! This module contains the `Normalizer` transformer.
//!
//! The `Normalizer` transformer is used to transform input data
//! so that the norm of each row is equal to 1. By default the
//! `Normalizer` uses the `Euclidean` norm.
//!
//! If input data has a row with all 0, `Normalizer` keeps the row as it is.
//!
//! Because transformation is performed per row independently,
//! inverse transformation is not supported.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::transforms::{Transformer, Normalizer};
//! use rusty_machine::linalg::Matrix;
//!
//! // Constructs a new `Normalizer`
//! let mut transformer = Normalizer::default();
//!
//! let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 1.5, 3.0]);
//!
//! // Transform the inputs
//! let transformed = transformer.transform(inputs).unwrap();
//! ```

use learning::error::{Error, ErrorKind};
use linalg::{BaseMatrix, BaseMatrixMut, Matrix, MatrixSlice};
use rulinalg::norm::{Euclidean, MatrixNorm};

use super::Transformer;

use libnum::Float;

use std::marker::PhantomData;

/// The Normalizer
///
/// The Normalizer provides an implementation of `Transformer`
/// which allows us to transform the all rows to have the same norm.
///
/// The default `Normalizer` will use the `Euclidean` norm.
///
/// See the module description for more information.
#[derive(Debug)]
pub struct Normalizer<T: Float, M>
where
    for<'a> M: MatrixNorm<T, MatrixSlice<'a, T>>,
{
    norm: M,
    _marker: PhantomData<T>,
}

/// Create a `Normalizer` with a Euclidean norm.
impl<T: Float> Default for Normalizer<T, Euclidean> {
    fn default() -> Self {
        Normalizer {
            norm: Euclidean,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, M> Normalizer<T, M>
where
    for<'a> M: MatrixNorm<T, MatrixSlice<'a, T>>,
{
    /// Constructs a new `Normalizer` with given norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::data::transforms::Normalizer;
    /// use rusty_machine::linalg::norm::Euclidean;
    ///
    /// // Constructs a new `Normalizer`
    /// let _ = Normalizer::<f64, Euclidean>::new(Euclidean);
    /// ```
    pub fn new(norm: M) -> Self {
        Normalizer {
            norm: norm,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, M> Transformer<Matrix<T>> for Normalizer<T, M>
where
    for<'a> M: MatrixNorm<T, MatrixSlice<'a, T>>,
{
    fn transform(&mut self, mut inputs: Matrix<T>) -> Result<Matrix<T>, Error> {
        let dists: Vec<T> = inputs.row_iter().map(|m| self.norm.norm(&*m)).collect();
        for (mut row, &d) in inputs.row_iter_mut().zip(dists.iter()) {
            if !d.is_finite() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Some data point is non-finite.",
                ));
            } else if d != T::zero() {
                // no change if distance is 0
                *row /= d;
            }
        }
        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Transformer;
    use super::*;
    use linalg::Matrix;

    use std::f64;

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::NAN; 4]);
        let mut normalizer = Normalizer::default();
        let res = normalizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn inf_data_test() {
        let inputs = Matrix::new(2, 2, vec![f64::INFINITY; 4]);
        let mut normalizer = Normalizer::default();
        let res = normalizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn single_row_test() {
        let inputs = matrix![1.0, 2.0];
        let mut normalizer = Normalizer::default();
        let transformed = normalizer.transform(inputs).unwrap();

        let exp = matrix![0.4472135954999579, 0.8944271909999159];
        assert_matrix_eq!(transformed, exp);
    }

    #[test]
    fn basic_normalizer_test() {
        let inputs = matrix![-1.0f32, 2.0;
                             0.0, 3.0];

        let mut normalizer = Normalizer::default();
        let transformed = normalizer.transform(inputs).unwrap();

        let exp = matrix![-0.4472135954999579, 0.8944271909999159;
                          0., 1.];
        assert_matrix_eq!(transformed, exp);

        let inputs = matrix![1., 2.;
                             10., 20.;
                             100., 200.];

        let transformed = normalizer.transform(inputs).unwrap();

        let exp = matrix![0.4472135954999579, 0.8944271909999159;
                          0.4472135954999579, 0.8944271909999159;
                          0.4472135954999579, 0.8944271909999159];
        assert_matrix_eq!(transformed, exp);

        let inputs = matrix![1., 2., 10.;
                             0., 10., 20.;
                             100., 10., 200.;
                             0., 0., 0.];
        let transformed = normalizer.transform(inputs).unwrap();

        let exp = matrix![0.09759000729485333, 0.19518001458970666, 0.9759000729485332;
                          0., 0.4472135954999579, 0.8944271909999159;
                          0.4467670516087703, 0.04467670516087703, 0.8935341032175406;
                          0., 0., 0.];
        assert_matrix_eq!(transformed, exp);
    }
}
