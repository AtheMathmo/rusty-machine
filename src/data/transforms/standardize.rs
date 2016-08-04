//! The standardizing transformer.

use learning::error::{Error, ErrorKind};
use linalg::{Matrix, Vector, Axes};
use super::Transformer;

use rulinalg::utils;

use libnum::{Float, FromPrimitive};

/// The standardizer
#[derive(Debug)]
pub struct Standardizer<T: Float> {
    /// Mins per column of input data
    means: Option<Vector<T>>,
    /// Maxs per column of input data
    variances: Option<Vector<T>>,
    /// The mean of the new data (default 0)
    scaled_mean: T,
    /// The standard deviation of the new data (default 1)
    scaled_stdev: T,
}

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
    /// Constructs a new Standardizer with the given mean and variance
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
                utils::in_place_vec_bin_op(row,
                                           &mean.data(),
                                           |x, &y| *x = *x - y);
                utils::in_place_vec_bin_op(row,
                                           &variance.data(),
                                           |x, &y| *x = (*x * self.scaled_stdev / y.sqrt()) + self.scaled_mean);
            }

            self.means = Some(mean);
            self.variances = Some(variance);
            Ok(inputs)
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
        let inputs = Matrix::new(1,2,vec![1.0, 2.0]);

        let mut standardizer = Standardizer::default();
        
        let res = standardizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn nan_data_test() {
        let inputs = Matrix::new(2,2,vec![f64::NAN; 4]);

        let mut standardizer = Standardizer::default();
        
        let res = standardizer.transform(inputs);
        assert!(res.is_err());
    }

    #[test]
    fn inf_data_test() {
        let inputs = Matrix::new(2,2,vec![f64::INFINITY; 4]);

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
}