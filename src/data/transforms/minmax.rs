//! The Min-Max transformer

use learning::error::{Error, ErrorKind};
use linalg::Matrix;
use super::Transformer;

use libnum::Float;

/// The min max scaler
#[derive(Debug)]
pub struct MinMaxScaler<T: Float> {
    /// Mins per column of input data
    scale_factors: Option<Vec<T>>,
    /// Maxs per column of input data
    const_factors: Option<Vec<T>>,
    /// The min of the new data (default 0)
    scaled_min: T,
    /// The max of the new data (default 1)
    scaled_max: T,
}

impl<T: Float> Default for MinMaxScaler<T> {
    fn default() -> MinMaxScaler<T> {
        MinMaxScaler {
            scale_factors: None,
            const_factors: None,
            scaled_min: T::zero(),
            scaled_max: T::one(),
        }
    }
}

impl<T: Float> MinMaxScaler<T> {
    /// Constructs a new MinMaxScaler with the specified scale.
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

        let mut input_min = vec![T::max_value(); features];
        let mut input_max = vec![T::min_value(); features];

        for row in inputs.iter_rows() {
            for (idx, feature) in row.into_iter().enumerate() {
                if !feature.is_finite() {
                    return Err(Error::new(ErrorKind::InvalidData,
                                          format!("Data point in column {} cannot be \
                                                   processed",
                                                  idx)));
                }

                if *feature < input_min[idx] {
                    input_min[idx] = *feature;
                }

                if *feature > input_max[idx] {
                    input_max[idx] = *feature;
                }
            }
        }

        // We'll scale each feature by a * x + b.
        // Where scales holds `a` per column and consts
        // holds `b`.
        let scales = input_min.iter()
            .zip(input_max.iter())
            .map(|(&x, &y)| (self.scaled_max - self.scaled_min) / (y - x))
            .collect::<Vec<_>>();

        if !scales.iter().all(|&x| x.is_finite()) {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "Constant feature columns not supported."));
        }

        let consts = input_max.iter()
            .zip(scales.iter())
            .map(|(&x, &s)| self.scaled_max - x * s)
            .collect::<Vec<_>>();

        for row in inputs.iter_rows_mut() {
            for i in 0..features {
                row[i] = scales[i] * row[i] + consts[i];
            }
        }

        self.scale_factors = Some(scales);
        self.const_factors = Some(consts);

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Transformer;
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

        // First column scales to zero and second to 1
        transformed[[0, 0]].abs() < 1e-10;
        transformed[[0, 1]].abs() < 1e-10;
        (transformed[[1, 0]] - 1.0).abs() < 1e-10;
        (transformed[[1, 1]] - 1.0).abs() < 1e-10;
    }

    #[test]
    fn constant_feature_test() {
        let inputs = Matrix::new(2, 2, vec![1.0, 2.0, 1.0, 3.0]);

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        let res = scaler.transform(inputs);

        assert!(res.is_err());
    }
}
