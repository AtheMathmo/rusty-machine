//! Gradient Descent
//!
//! Implementation of gradient descent algorithm. Module contains
//! the struct `GradientDesc` which is instantiated within models
//! implementing the Optimizable trait.
//!
//! Currently standard batch gradient descent is the only implemented
//! optimization algorithm but there is flexibility to introduce new
//! algorithms and git them into the same scheme easily.

use learning::optim::{Optimizable, OptimAlgorithm};
use linalg::Vector;
use linalg::Matrix;
use rulinalg::utils;

use learning::toolkit::rand_utils;

const LEARNING_EPS: f64 = 1e-20;

/// Batch Gradient Descent algorithm
#[derive(Clone, Copy, Debug)]
pub struct GradientDesc {
    /// The step-size for the gradient descent steps.
    alpha: f64,
    /// The number of iterations to run.
    iters: usize,
}

/// The default gradient descent algorithm.
///
/// The defaults are:
///
/// - alpha = 0.3
/// - iters = 100
impl Default for GradientDesc {
    fn default() -> GradientDesc {
        GradientDesc {
            alpha: 0.3,
            iters: 100,
        }
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
        assert!(alpha > 0f64,
                "The step size (alpha) must be greater than 0.");

        GradientDesc {
            alpha: alpha,
            iters: iters,
        }
    }
}

impl<M: Optimizable> OptimAlgorithm<M> for GradientDesc {
    fn optimize(&self,
                model: &M,
                start: &[f64],
                inputs: &M::Inputs,
                targets: &M::Targets)
                -> Vec<f64> {

        // Create the initial optimal parameters
        let mut optimizing_val = Vector::new(start.to_vec());
        // The cost at the start of each iteration
        let mut start_iter_cost = 0f64;

        for _ in 0..self.iters {
            // Compute the cost and gradient for the current parameters
            let (cost, grad) = model.compute_grad(optimizing_val.data(), inputs, targets);

            // Early stopping
            if (start_iter_cost - cost).abs() < LEARNING_EPS {
                break;
            } else {
                // Update the optimal parameters using gradient descent
                optimizing_val = &optimizing_val - Vector::new(grad) * self.alpha;
                // Update the latest cost
                start_iter_cost = cost;
            }
        }
        optimizing_val.into_vec()
    }
}

/// Stochastic Gradient Descent algorithm.
///
/// Uses basic momentum to control the learning rate.
#[derive(Clone, Copy, Debug)]
pub struct StochasticGD {
    /// Controls the momentum of the descent
    alpha: f64,
    /// The square root of the raw learning rate.
    mu: f64,
    /// The number of passes through the data.
    iters: usize,
}

/// The default Stochastic GD algorithm.
///
/// The defaults are:
///
/// - alpha = 0.1
/// - mu = 0.1
/// - iters = 20
impl Default for StochasticGD {
    fn default() -> StochasticGD {
        StochasticGD {
            alpha: 0.1,
            mu: 0.1,
            iters: 20,
        }
    }
}

impl StochasticGD {
    /// Construct a stochastic gradient descent algorithm.
    ///
    /// Requires the learning rate, momentum rate and iteration count
    /// to be specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// let sgd = StochasticGD::new(0.1, 0.3, 5);
    /// ```
    pub fn new(alpha: f64, mu: f64, iters: usize) -> StochasticGD {
        assert!(alpha > 0f64, "The momentum (alpha) must be greater than 0.");
        assert!(mu > 0f64, "The step size (mu) must be greater than 0.");

        StochasticGD {
            alpha: alpha,
            mu: mu,
            iters: iters,
        }
    }
}

impl<M> OptimAlgorithm<M> for StochasticGD
    where M: Optimizable<Inputs = Matrix<f64>, Targets = Matrix<f64>>
{
    fn optimize(&self,
                model: &M,
                start: &[f64],
                inputs: &M::Inputs,
                targets: &M::Targets)
                -> Vec<f64> {

        // Create the initial optimal parameters
        let mut optimizing_val = Vector::new(start.to_vec());
        // Create the momentum based gradient distance
        let mut delta_w = Vector::zeros(start.len());

        // Set up the indices for permutation
        let mut permutation = (0..inputs.rows()).collect::<Vec<_>>();
        // The cost at the start of each iteration
        let mut start_iter_cost = 0f64;

        for _ in 0..self.iters {
            // The cost at the end of each stochastic gd pass
            let mut end_cost = 0f64;
            // Permute the indices
            rand_utils::in_place_fisher_yates(&mut permutation);
            for i in &permutation {
                // Compute the cost and gradient for this data pair
                let (cost, vec_data) = model.compute_grad(optimizing_val.data(),
                                                          &inputs.select_rows(&[*i]),
                                                          &targets.select_rows(&[*i]));

                // Compute the difference in gradient using momentum
                delta_w = Vector::new(vec_data) * self.mu + &delta_w * self.alpha;
                // Update the parameters
                optimizing_val = &optimizing_val - &delta_w * self.mu;
                // Set the end cost (this is only used after the last iteration)
                end_cost += cost;
            }

            end_cost /= inputs.rows() as f64;

            // Early stopping
            if (start_iter_cost - end_cost).abs() < LEARNING_EPS {
                break;
            } else {
                // Update the cost
                start_iter_cost = end_cost;
            }
        }
        optimizing_val.into_vec()
    }
}

/// Adaptive Gradient Descent
///
/// The adaptive gradient descent algorithm (Duchi et al. 2010).
#[derive(Debug)]
pub struct AdaGrad {
    alpha: f64,
    tau: f64,
    iters: usize,
}

impl AdaGrad {
    /// Constructs a new AdaGrad algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::optim::grad_desc::AdaGrad;
    ///
    /// // Create a new AdaGrad algorithm with step size 0.5
    /// // and adaptive scaling constant 1.0
    /// let gd = AdaGrad::new(0.5, 1.0, 100);
    /// ```
    pub fn new(alpha: f64, tau: f64, iters: usize) -> AdaGrad {
        assert!(alpha > 0f64,
                "The step size (alpha) must be greater than 0.");
        assert!(tau >= 0f64,
                "The adaptive constant (tau) cannot be negative.");
        AdaGrad {
            alpha: alpha,
            tau: tau,
            iters: iters,
        }
    }
}

impl Default for AdaGrad {
    fn default() -> AdaGrad {
        AdaGrad {
            alpha: 1f64,
            tau: 3f64,
            iters: 100,
        }
    }
}

impl<M: Optimizable<Inputs = Matrix<f64>, Targets = Matrix<f64>>> OptimAlgorithm<M> for AdaGrad {
    fn optimize(&self,
                model: &M,
                start: &[f64],
                inputs: &M::Inputs,
                targets: &M::Targets)
                -> Vec<f64> {

        // Initialize the adaptive scaling
        let mut ada_s = Vector::zeros(start.len());
        // Initialize the optimal parameters
        let mut optimizing_val = Vector::new(start.to_vec());

        // Set up the indices for permutation
        let mut permutation = (0..inputs.rows()).collect::<Vec<_>>();
        // The cost at the start of each iteration
        let mut start_iter_cost = 0f64;

        for _ in 0..self.iters {
            // The cost at the end of each stochastic gd pass
            let mut end_cost = 0f64;
            // Permute the indices
            rand_utils::in_place_fisher_yates(&mut permutation);
            for i in &permutation {
                // Compute the cost and gradient for this data pair
                let (cost, mut vec_data) = model.compute_grad(optimizing_val.data(),
                                                              &inputs.select_rows(&[*i]),
                                                              &targets.select_rows(&[*i]));
                // Update the adaptive scaling by adding the gradient squared
                utils::in_place_vec_bin_op(ada_s.mut_data(), &vec_data, |x, &y| *x += y * y);

                // Compute the change in gradient
                utils::in_place_vec_bin_op(&mut vec_data, ada_s.data(), |x, &y| {
                    *x = self.alpha * (*x / (self.tau + (y).sqrt()))
                });
                // Update the parameters
                optimizing_val = &optimizing_val - Vector::new(vec_data);
                // Set the end cost (this is only used after the last iteration)
                end_cost += cost;
            }
            end_cost /= inputs.rows() as f64;

            // Early stopping
            if (start_iter_cost - end_cost).abs() < LEARNING_EPS {
                break;
            } else {
                // Update the cost
                start_iter_cost = end_cost;
            }
        }
        optimizing_val.into_vec()
    }
}

#[cfg(test)]
mod tests {

    use super::{GradientDesc, StochasticGD, AdaGrad};

    #[test]
    #[should_panic]
    fn gd_neg_stepsize() {
        let _ = GradientDesc::new(-0.5, 0);
    }

    #[test]
    #[should_panic]
    fn stochastic_gd_neg_momentum() {
        let _ = StochasticGD::new(-0.5, 1f64, 0);
    }

    #[test]
    #[should_panic]
    fn stochastic_gd_neg_stepsize() {
        let _ = StochasticGD::new(0.5, -1f64, 0);
    }

    #[test]
    #[should_panic]
    fn adagrad_neg_stepsize() {
        let _ = AdaGrad::new(-0.5, 1f64, 0);
    }

    #[test]
    #[should_panic]
    fn adagrad_neg_adaptive_scale() {
        let _ = AdaGrad::new(0.5, -1f64, 0);
    }

}
