//! Adam Optimizer
//! 
//! Implementation of the ADAM optimization algorithm.
//! 
use learning::optim::{Optimizable, OptimAlgorithm};
use linalg::Vector;
use linalg::{Matrix, BaseMatrix};
use rulinalg::utils;

use learning::toolkit::rand_utils;

const EVAL_STEP: usize = 10;

/// Adam Optimizer
#[derive(Debug)]
pub struct Adam {
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    iters: usize
}


impl Adam {
    /// Construct an Adam algorithm.
    ///
    /// Requires learning rate, exponential decay rates, epsilon, and iteration count.
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, iters: usize) -> Adam {
        assert!(0f64 < learning_rate, "The learning rate must be positive");
        assert!((0f64 <= beta1 && beta1 < 1f64), "Beta value be within the range of [0,1)");
        assert!((0f64 <= beta2 && beta2 < 1f64), "Beta value be within the range of [0,1)");
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

/// The default ADAM configuration
///
/// The defaults are:
///
/// - alpha = 0.001 (lr)
/// - beta1 = 0.09 (dw)
/// - beta2 = 0.999 (dw^2)
/// - epsilon = 1e-8
/// - iters = 50
/// source: https://arxiv.org/pdf/1412.6980.pdf
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

impl<M> OptimAlgorithm<M> for Adam
    where M: Optimizable<Inputs = Matrix<f64>, Targets = Matrix<f64>> {
    fn optimize(&self,
                model: &M,
                start: &[f64],
                inputs: &M::Inputs,
                targets: &M::Targets) 
                -> Vec<f64> {
        // Initial parameters
        let mut params = Vector::new(start.to_vec());

        // Set up the indices for permutation
        let mut permutation = (0..inputs.rows()).collect::<Vec<_>>();

        // moment vectors & timestep
        let mut t: f64 = 0.0;
        let mut m = Vector::zeros(start.len());
        let mut v = Vector::zeros(start.len());


        let mut loss_vector: Vec<f64> = vec![];

        for l in 0..self.iters {
            // The cost at the end of each pass

            if l % EVAL_STEP == 0 && l > 0 {
                let average_loss: f64 = loss_vector.iter().sum::<f64>() / loss_vector.len() as f64;
                println!("Running average loss iter {:#?}: {:#?}", l, average_loss);
            }

            // Permute the indices
            rand_utils::in_place_fisher_yates(&mut permutation);
            for i in &permutation {
                // Incrementing the time step
                t += 1.0;
                // Comput the cost and gradient
                let (cost, grad) = model.compute_grad(params.data(),
                                                      &inputs.select_rows(&[*i]),
                                                      &targets.select_rows(&[*i]));

                let grad = Vector::new(grad);
                let grad_squared = grad.clone().apply(&|x| x * x);

                //Moving averages of the gradients
                m = m * self.beta1 + grad * (1.0 - self.beta1);

                // Moving averages of the squared gradients
                v = v * self.beta2 + grad_squared * (1.0 - self.beta2);

                // Bias-corrected estimates
                let mut m_hat = &m / (1.0 - (self.beta1.powf(t)));
                let mut v_hat = &v / (1.0 - (self.beta2.powf(t)));

                utils::in_place_vec_bin_op(m_hat.mut_data(), v_hat.data(), |x, &y| {
                    *x = (*x / &y.sqrt() - self.epsilon) * self.alpha;
                });

                // update params
                params = &params - &m_hat;

                loss_vector.push(cost);
            }
        }
        params.into_vec()
    }

}