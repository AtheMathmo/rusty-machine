use linalg::matrix::Matrix;
use learning::optim::OptimAlgorithm;

struct GradientDesc {
	alpha: f64,
	iters: usize
}

impl OptimAlgorithm<Matrix<f64>> for GradientDesc {
	fn optimize(&self, start: Matrix<f64>, f: &Fn(Matrix<f64>) -> (f64, Matrix<f64>)) -> Matrix<f64> {

		let mut optimizing_val = start.clone();

		for _i in 0..self.iters {
			optimizing_val = optimizing_val.clone() - f(optimizing_val.clone()).1 * self.alpha;
		}
		optimizing_val
	}
} 