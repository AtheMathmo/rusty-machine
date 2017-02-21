//! The Shuffler
//!
//! This module contains the `Shuffler` transformer. `Shuffler` implements the
//! `Transformer` trait and is used to shuffle the rows of an input matrix.
//! You can control the random number generator used by the `Shuffler`.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::data::transforms::Transformer;
//! use rusty_machine::data::transforms::shuffle::Shuffler;
//!
//! // Create an input matrix that we want to shuffle
//! let mat = Matrix::new(3, 2, vec![1.0, 2.0,
//!                                  3.0, 4.0,
//!                                  5.0, 6.0]);
//!
//! // Create a new shuffler
//! let mut shuffler = Shuffler::default();
//! let shuffled_mat = shuffler.transform(mat).unwrap();
//!
//! println!("{}", shuffled_mat);
//! ```

use learning::LearningResult;
use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use super::Transformer;

use rand::{Rng, thread_rng, ThreadRng};

/// The `Shuffler`
///
/// Provides an implementation of `Transformer` which shuffles
/// the input rows in place.
#[derive(Debug)]
pub struct Shuffler<R: Rng> {
    rng: R,
}

impl<R: Rng> Shuffler<R> {
    /// Construct a new `Shuffler` with given random number generator.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate rusty_machine;
    ///
    /// use rusty_machine::data::transforms::Transformer;
    /// use rusty_machine::data::transforms::shuffle::Shuffler;
    /// use rand::{StdRng, SeedableRng};
    ///
    /// # fn main() {
    /// // We can create a seeded rng
    /// let rng = StdRng::from_seed(&[1, 2, 3]);
    ///
    /// let shuffler = Shuffler::new(rng);
    /// # }
    /// ```
    pub fn new(rng: R) -> Self {
        Shuffler { rng: rng }
    }
}

/// Create a new shuffler using the `rand::thread_rng` function
/// to provide a randomly seeded random number generator.
impl Default for Shuffler<ThreadRng> {
    fn default() -> Self {
        Shuffler { rng: thread_rng() }
    }
}

/// The `Shuffler` will transform the input `Matrix` by shuffling
/// its rows in place.
///
/// Under the hood this uses a Fisher-Yates shuffle.
impl<R: Rng, T> Transformer<Matrix<T>> for Shuffler<R> {
    fn transform(&mut self, mut inputs: Matrix<T>) -> LearningResult<Matrix<T>> {
        let n = inputs.rows();

        for i in 0..n {
            // Swap i with a random point after it
            let j = self.rng.gen_range(0, n - i);
            inputs.swap_rows(i, i + j);
        }
        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use linalg::Matrix;
    use super::super::Transformer;
    use super::Shuffler;

    use rand::{StdRng, SeedableRng};

    #[test]
    fn seeded_shuffle() {
        let rng = StdRng::from_seed(&[1, 2, 3]);
        let mut shuffler = Shuffler::new(rng);

        let mat = Matrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let shuffled = shuffler.transform(mat).unwrap();

        assert_eq!(shuffled.into_vec(),
                   vec![3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]);
    }

    #[test]
    fn shuffle_single_row() {
        let mut shuffler = Shuffler::default();

        let mat = Matrix::new(1, 8, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let shuffled = shuffler.transform(mat).unwrap();

        assert_eq!(shuffled.into_vec(),
                   vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}