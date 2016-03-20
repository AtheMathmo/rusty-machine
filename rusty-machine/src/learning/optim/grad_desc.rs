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
use linalg::matrix::Matrix;

/// Batch Gradient Descent algorithm
#[derive(Clone, Copy, Debug)]
pub struct GradientDesc {
    /// The step-size for the gradient descent steps.
    pub alpha: f64,
    /// The number of iterations to run.
    pub iters: usize,
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

        let mut optimizing_val = Vector::new(start.to_vec());

        for _ in 0..self.iters {
            optimizing_val = &optimizing_val -
                             Vector::new(model.compute_grad(&optimizing_val.data()[..],
                                                            inputs,
                                                            targets)
                                              .1) * self.alpha;
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
    pub alpha: f64,
    /// The square root of the raw learning rate.
    pub mu: f64,
    /// The number of passes through the data.
    pub iters: usize,
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
        StochasticGD {
            alpha: alpha,
            mu: mu,
            iters: iters,
        }
    }
}

impl<M: Optimizable<Inputs = Matrix<f64>, Targets = Matrix<f64>>> OptimAlgorithm<M> for StochasticGD {

    fn optimize(&self, model: &M, start: &[f64], inputs: &M::Inputs, targets: &M::Targets) -> Vec<f64> {

        let (_, vec_data) = model.compute_grad(start,
                                               &inputs.select_rows(&[0]),
                                               &targets.select_rows(&[0]));
        let grad = Vector::new(vec_data);
        let mut delta_w = grad * self.alpha;
        let mut optimizing_val = Vector::new(start.to_vec()) - &delta_w * self.mu;

        for _ in 0..self.iters {
            for i in 1..inputs.rows() {
                let (_, vec_data) = model.compute_grad(&optimizing_val.data()[..],
                                                       &inputs.select_rows(&[i]),
                                                       &targets.select_rows(&[i]));

                delta_w = Vector::new(vec_data) * self.mu + &delta_w * self.alpha;
                optimizing_val = &optimizing_val - &delta_w * self.mu;
            }
        }
        optimizing_val.into_vec()
    }
}
