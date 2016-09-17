//! The Shuffler

/// The Shuffler
pub struct Shuffler;

use learning::error::{Error, ErrorKind};
use learning::LearningResult;
use linalg::{Matrix, Vector, Axes};
use super::Transformer;

use rand::{Rng, thread_rng};
use rulinalg::utils;

impl<T> Transformer<Matrix<T>> for Shuffler {
    /// Transforms the inputs and stores the transformation in the Transformer
    fn transform(&mut self, inputs: Matrix<T>) -> LearningResult<Matrix<T>> {
        let n = inputs.rows();
        let mut rng = thread_rng();

        for i in 0..n {
            // Swap i with a random point after it
            let j = rng.gen_range(0, n - i);
            inputs.swap_rows(i, i + j);
        }

        Ok(inputs)
    }
}