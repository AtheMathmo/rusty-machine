//! Gradient Descent
//! 
//! Implementation of gradient descent algorithm. Module contains
//! the struct GradientDesc which is instantiated within models
//! implementing the Optimizable trait.
//!
//! Currently standard batch gradient descent is the only implemented
//! optimization algorithm but there is flexibility to introduce new
//! algorithms and git them into the same scheme easily.

use learning::optim::{Optimizable, OptimAlgorithm};
use linalg::vector::Vector;

pub struct GradientDesc {
	alpha: f64,
	iters: usize
}

impl GradientDesc {

	/// Constructs a new gradient descent algorithm.
	///
	/// Currently defaults to 1000 iterations and step size
	/// of 0.3. More control coming soon!
	pub fn new() -> GradientDesc {
		GradientDesc{ alpha: 0.3, iters: 1000 }
	}
}

// This could take in the model to be optimized, which has an optimizable trait.
// This trait specifies a function which we can call within this trait.
impl<M: Optimizable> OptimAlgorithm<M> for GradientDesc {
    fn optimize(&self, model: &M, start: &[f64], data: &M::Data, outputs: &M::Target) -> Vec<f64> {

		let mut optimizing_val = Vector::new(start.to_vec());

		for _i in 0..self.iters {
			optimizing_val = optimizing_val.clone() - Vector::new(model.compute_grad(&optimizing_val.data[..], data, outputs)) * self.alpha;
		}
		optimizing_val.data
	}
} 