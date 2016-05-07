use linalg::Metric;
use linalg::matrix::{Matrix, MatrixSlice};
use libnum::Float;

/// Model Regularization
#[derive(Debug, Clone, Copy)]
pub enum Regularization<T: Float> {
    /// L2 Regularization
    L2(T),
}

impl<T : Float> Regularization<T> {

	/// Compute the regularization addition to the cost.
	pub fn reg_cost(&self, mat: MatrixSlice<T>) -> T {
		match self {
			&Regularization::L2(x) => mat.norm() * x / (T::one() + T::one()),
		}
	}

	/// Compute the regularization addition to the gradient.
	pub fn reg_grad(&self, mat: MatrixSlice<T>) -> Matrix<T> {
		match self {
			&Regularization::L2(x) => mat * x,
		}
	}
} 