pub trait CostFunc<T> {
	fn cost(output: &T, target: &T) -> f64;

	fn grad_cost(output: &T, target: &T) -> T;
}

struct MeanSqError;

// We need a trait for "Hadamard product" here
// Which is "Elementwise multiplication".
impl<Matrix<f64>> CostFunc<Matrix<f64>> for MeanSqError {
	fn cost(output: &T, target: &T) -> f64 {
		// The cost for a single
		diff = output - target
		let sq_diff = &diff.elemul(&diff);

		let n = diff.rows();

		sq_diff.sum() / (2 * n)
	}

	fn grad_cost(output: &T, target: &T) -> T {
		output - target
	}
}