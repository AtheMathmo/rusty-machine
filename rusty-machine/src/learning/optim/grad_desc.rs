use std::ops::{Mul, Sub};
use learning::optim::OptimAlgorithm;

pub struct GradientDesc {
	alpha: f64,
	iters: usize
}

impl<T: Clone + Mul<f64, Output=T> + Sub<T, Output=T>> OptimAlgorithm<T> for GradientDesc {
	fn optimize(&self, start: T, f: &Fn(T) -> (f64, T)) -> T {

		let mut optimizing_val = start.clone();

		for _i in 0..self.iters {
			optimizing_val = optimizing_val.clone() - f(optimizing_val.clone()).1 * self.alpha;
		}
		optimizing_val
	}
} 