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
	pub alpha: f64,
	pub iters: usize
}

impl Default for GradientDesc {

	/// Constructs a gradient descent algorithm
	/// with default settings.
	///
	/// Uses 10000 iterations and step size of 0.3.
	fn default() -> GradientDesc {
		GradientDesc{ alpha: 0.3, iters: 10000 }
	}
}

impl GradientDesc {

	/// Construct a gradient descent algorithm.
	///
	/// Requires the step size and iteration count
	/// to be specified.
	///
	/// # Examples
	///
	/// ```
	/// use rusty_machine::learning::optim::grad_desc::GradientDesc;
	///
	/// let gd = GradientDesc::new(0.3, 10000);
	/// ```
	pub fn new(alpha: f64, iters: usize) -> GradientDesc {
		GradientDesc{ alpha: alpha, iters: iters }
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