use learning::optim::{Optimizable, OptimAlgorithm};
use linalg::Vector;
use linalg::{Matrix, BaseMatrix};
use rulinalg::utils;

use learning::toolkit::rand_utils;

// Adam Optimizer
pub struct Adam {
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    iters: usize
}

// The default ADAM configuration
//
// The defaults are:
//
// - alpha = 0.001 (lr)
// - beta1 = 0.09 (dw)
// - beta2 = 0.999 (dw^2)
// - epsilon = 1e-8
// - iters = 50
// source: https://arxiv.org/pdf/1412.6980.pdf
impl Default for Adam {
    fn default() -> Adam {
        Adam {
            alpha: 0.001,
            beta1: 0.09,
            beta2: 0.999,
            epsilon: 1e-8,
            iters: 50
        }
    }
}

impl Adam {
    // Construct an Adam algorithm.
    //
    // Requires learning rate, exponential decay rates, epsilon, and iteration count.
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, iters: usize) -> Adam {
        assert!(0f64 < learning_rate, "The learning rate must be positive");
        assert!(0f64 <= beta1 < 1, "Beta value be within the range of [0,1)");
        assert!(0f64 <= beta2 < 1, "Beta value be within the range of [0,1)");
        assert!(0f64 < epsilon, "Epsilon must be positive");

        Adam {
            alpha: learning_rate,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            iters: iters
        }
    }
}

impl<M> OptimAlgorithm<M> for Adam
    where M: Optimizable<inputs = Matrix<f64>, Targets = Matrix<f64>> {
    fn optimize(&self,
                model: &M,
                start: &[f64],
                inputs: &M::Inputs,
                targets: &M::Targets) 
                -> Vec<f64> {
        // Initial parameters
        let mut params: f64 = Vector::new(start.to_vec());

        // moment 
        let mut m, v, t = 0f64;
    }

}