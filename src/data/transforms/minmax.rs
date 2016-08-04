//! The Min-Max transformer

use learning::error::{Error, ErrorKind};
use linalg::Matrix;
use super::Transformer;
use std::f64;

/// The min max scaler
#[derive(Debug)]
pub struct MinMaxScaler {
    /// Mins per column of input data
    input_min: Vec<f64>,
    /// Maxs per column of input data
    input_max: Vec<f64>,
    /// The min of the new data (default 0)
    scaled_min: f64,
    /// The max of the new data (default 1)
    scaled_max: f64,
}

impl MinMaxScaler {
    /// Constructs a new MinMaxScaler with the specified scale.
    pub fn new(min: f64, max: f64) -> MinMaxScaler {
        MinMaxScaler {
            input_min: Vec::new(),
            input_max: Vec::new(),
            scaled_min: min,
            scaled_max: max,
        }
    }
}

impl Transformer<Matrix<f64>> for MinMaxScaler {
    fn transform(&mut self, mut inputs: Matrix<f64>) -> Result<Matrix<f64>, Error> {
        let features = inputs.cols();

        self.input_min = vec![f64::MAX; features];
        self.input_max = vec![f64::MIN; features];

        for row in inputs.iter_rows() {
            for (idx, feature) in row.into_iter().enumerate() {
                if !feature.is_finite() {
                    return Err(Error::new(ErrorKind::InvalidData,
                                          format!("Data point {0} in column {1} cannot be \
                                                   processed",
                                                  feature,
                                                  idx)));
                }

                if *feature < self.input_min[idx] {
                    self.input_min[idx] = *feature;
                }

                if *feature > self.input_max[idx] {
                    self.input_max[idx] = *feature;
                }
            }
        }

        // We'll scale each feature by a * x + b.
        // Where scales holds `a` per column and consts
        // holds `b`.
        let scales = self.input_min
            .iter()
            .zip(self.input_max.iter())
            .map(|(&x, &y)| (self.scaled_max - self.scaled_min) / (y - x))
            .collect::<Vec<_>>();

        let consts = self.input_max
            .iter()
            .zip(scales.iter())
            .map(|(&x, &s)| self.scaled_max - x * s)
            .collect::<Vec<_>>();

        for row in inputs.iter_rows_mut() {
            for (idx, feature) in row.into_iter().enumerate() {
                *feature = scales[idx] * *feature + consts[idx];
            }
        }

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
        let inputs = Matrix::new(2, 2, vec![-1.0, 2.0, 0.0, 3.0]);

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        let transformed = scaler.transform(inputs).unwrap();

        println!("{}", transformed);

        assert!(transformed.data().iter().all(|&x| x >= 0.0));
        assert!(transformed.data().iter().all(|&x| x <= 1.0));
    }
}
