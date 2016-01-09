use linalg::matrix::Matrix;

pub trait CostFunc<T> {
	fn cost(output: &T, target: &T) -> f64;

	fn grad_cost(output: &T, target: &T) -> T;
}

pub struct MeanSqError;

// We need a trait for "Hadamard product" here
// Which is "Elementwise multiplication".
impl CostFunc<Matrix<f64>> for MeanSqError {
	fn cost(output: &Matrix<f64>, target: &Matrix<f64>) -> f64 {
		// The cost for a single
		let diff = output - target;
		let sq_diff = &diff.elemul(&diff);

		let n = diff.rows();

		sq_diff.sum() / (2f64 * (n as f64))
	}

	fn grad_cost(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
		output - target
	}
}

pub struct CrossEntropyError;

impl CostFunc<Matrix<f64>> for CrossEntropyError {
	fn cost(output: &Matrix<f64>, target: &Matrix<f64>) -> f64 {
		// The cost for a single
		let log_inv_output = (-output + 1f64).apply(&ln);
		let log_output = output.clone().apply(&ln);

		let mat_cost = target.elemul(&log_output) + (-target+1f64).elemul(&log_inv_output);

		let n = output.rows();

		- (mat_cost.sum()) / (n as f64)
	}

	fn grad_cost(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
		(output - target).elediv(&(output.elemul(&(-output+1f64))))
	}
}

fn ln(x: f64) -> f64 {
	x.ln()
}