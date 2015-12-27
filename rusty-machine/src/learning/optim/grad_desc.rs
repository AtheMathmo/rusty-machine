use learning::optim::{OptimAlgorithm, Optimizable};
use linalg::vector::Vector;

pub struct GradientDesc {
	alpha: f64,
	iters: usize
}

// This could take in the model to be optimized, which has an optimizable trait.
// This trait specifies a function which we can call within this trait.
impl<M: Optimizable<T,U>, T, U> OptimAlgorithm<T,U, M> for GradientDesc {
    fn optimize(&self, model: M, start: &[f64], data: &T, output: &U) -> Vec<f64> {

		let mut optimizing_val = Vector::new(start.to_vec());

		for _i in 0..self.iters {
			optimizing_val = optimizing_val.clone() - Vector::new(model.compute_grad(&optimizing_val.data[..], data, output)) * self.alpha;
		}
		optimizing_val.data
	}
} 