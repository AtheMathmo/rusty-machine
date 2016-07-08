extern crate num as libnum;
extern crate matrixmultiply;

pub mod matrix;
pub mod convert;
pub mod macros;
pub mod error;
pub mod utils;
pub mod vector;

/// Trait for linear algebra metrics.
///
/// Currently only implements basic euclidean norm.
pub trait Metric<T> {
    /// Computes the euclidean norm.
    fn norm(&self) -> T;
}
