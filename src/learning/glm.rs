//! Generalized Linear Model module
//!
//! <i>This model is likely to undergo changes in the near future.
//! These changes will improve the learning algorithm.</i>
//!
//! Contains implemention of generalized linear models using
//! iteratively reweighted least squares.
//!
//! The model will automatically add the intercept term to the
//! input data.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::glm::{GenLinearModel, Bernoulli};
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//!
//! let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
//! let targets = Vector::new(vec![0.,0.,1.,1.]);
//!
//! // Construct a GLM with a Bernoulli distribution
//! // This is equivalent to a logistic regression model.
//! let mut log_mod = GenLinearModel::new(Bernoulli);
//!
//! // Train the model
//! log_mod.train(&inputs, &targets).unwrap();
//!
//! // Now we'll predict a new point
//! let new_point = Matrix::new(1,1,vec![10.]);
//! let output = log_mod.predict(&new_point).unwrap();
//!
//! // Hopefully we classified our new point correctly!
//! assert!(output[0] > 0.5, "Our classifier isn't very good!");
//! ```

use linalg::Vector;
use linalg::{Matrix, BaseMatrix};

use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};

/// The Generalized Linear Model
///
/// The model is generic over a Criterion
/// which specifies the distribution family and
/// the link function.
#[derive(Debug)]
pub struct GenLinearModel<C: Criterion> {
    parameters: Option<Vector<f64>>,
    criterion: C,
}

impl<C: Criterion> GenLinearModel<C> {
    /// Constructs a new Generalized Linear Model.
    ///
    /// Takes a Criterion which fully specifies the family
    /// and the link function used by the GLM.
    ///
    /// ```
    /// use rusty_machine::learning::glm::GenLinearModel;
    /// use rusty_machine::learning::glm::Bernoulli;
    ///
    /// let glm = GenLinearModel::new(Bernoulli);
    /// ```
    pub fn new(criterion: C) -> GenLinearModel<C> {
        GenLinearModel {
            parameters: None,
            criterion: criterion,
        }
    }
}

/// Supervised model trait for the GLM.
///
/// Predictions are made from the model by computing g^-1(Xb).
///
/// The model is trained using Iteratively Re-weighted Least Squares.
impl<C: Criterion> SupModel<Matrix<f64>, Vector<f64>> for GenLinearModel<C> {
    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<f64>> {
        if let Some(ref v) = self.parameters {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            Ok(self.criterion.apply_link_inv(full_inputs * v))
        } else {
            Err(Error::new_untrained())
        }
    }

    /// Train the model using inputs and targets.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) -> LearningResult<()> {
        let n = inputs.rows();

        if n != targets.size() {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "Training data do not have the same dimensions"));
        }

        // Construct initial estimate for mu
        let mut mu = Vector::new(self.criterion.initialize_mu(targets.data()));
        let mut z = mu.clone();
        let mut beta: Vector<f64> = Vector::new(vec![0f64; inputs.cols() + 1]);

        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);
        let x_t = full_inputs.transpose();

        // Iterate to convergence
        for _ in 0..8 {
            let w_diag = self.criterion.compute_working_weight(mu.data());
            let y_bar_data = self.criterion.compute_y_bar(targets.data(), mu.data());

            let w = Matrix::from_diag(&w_diag);
            let y_bar = Vector::new(y_bar_data);

            let x_t_w = &x_t * w;

            let new_beta = (&x_t_w * &full_inputs)
                .inverse()
                .expect("Could not compute input data inverse.") *
                           x_t_w * z;
            let diff = (beta - &new_beta).apply(&|x| x.abs()).sum();
            beta = new_beta;

            if diff < 1e-10 {
                break;
            }

            // Update z and mu
            let fitted = &full_inputs * &beta;
            z = y_bar + &fitted;
            mu = self.criterion.apply_link_inv(fitted);
        }

        self.parameters = Some(beta);
        Ok(())
    }
}

/// The criterion for the Generalized Linear Model.
///
/// This trait specifies a Link function and requires a model
/// variance to be specified. The model variance must be defined
/// to specify the regression family. The other functions need not
/// be specified but can be used to control optimization.
pub trait Criterion {
    /// The link function of the GLM Criterion.
    type Link: LinkFunc;

    /// The variance of the regression family.
    fn model_variance(&self, mu: f64) -> f64;

    /// Initializes the mean value.
    ///
    /// By default the mean takes the training target values.
    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        y.to_vec()
    }

    /// Computes the working weights that make up the diagonal
    /// of the `W` matrix used in the iterative reweighted least squares
    /// algorithm.
    ///
    /// This is equal to:
    ///
    /// 1 / (Var(u) * g'(u) * g'(u))
    fn compute_working_weight(&self, mu: &[f64]) -> Vec<f64> {
        let mut working_weights_vec = Vec::with_capacity(mu.len());

        for m in mu {
            let grad = self.link_grad(*m);
            working_weights_vec.push(1f64 / (self.model_variance(*m) * grad * grad));
        }

        working_weights_vec
    }

    /// Computes the adjustment to the fitted values used during
    /// fitting.
    ///
    /// This is equal to:
    ///
    /// g`(u) * (y - u)
    fn compute_y_bar(&self, y: &[f64], mu: &[f64]) -> Vec<f64> {
        let mut y_bar_vec = Vec::with_capacity(mu.len());

        for (idx, m) in mu.iter().enumerate() {
            y_bar_vec.push(self.link_grad(*m) * (y[idx] - m));
        }

        y_bar_vec
    }

    /// Applies the link function to a vector.
    fn apply_link_func(&self, vec: Vector<f64>) -> Vector<f64> {
        vec.apply(&Self::Link::func)
    }

    /// Applies the inverse of the link function to a vector.
    fn apply_link_inv(&self, vec: Vector<f64>) -> Vector<f64> {
        vec.apply(&Self::Link::func_inv)
    }

    /// Computes the gradient of the link function.
    fn link_grad(&self, mu: f64) -> f64 {
        Self::Link::func_grad(mu)
    }
}

/// Link functions.
///
/// Used within Generalized Linear Regression models.
pub trait LinkFunc {
    /// The link function.
    fn func(x: f64) -> f64;

    /// The gradient of the link function.
    fn func_grad(x: f64) -> f64;

    /// The inverse of the link function.
    /// Often called the 'mean' function.
    fn func_inv(x: f64) -> f64;
}

/// The Logit link function.
///
/// Used primarily as the canonical link in Binomial Regression.
#[derive(Clone, Copy, Debug)]
pub struct Logit;

/// The Logit link function.
///
/// g(u) = ln(x / (1 - x))
impl LinkFunc for Logit {
    fn func(x: f64) -> f64 {
        (x / (1f64 - x)).ln()
    }

    fn func_grad(x: f64) -> f64 {
        1f64 / (x * (1f64 - x))
    }

    fn func_inv(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// The log link function.
///
/// Used primarily as the canonical link in Poisson Regression.
#[derive(Clone, Copy, Debug)]
pub struct Log;

/// The log link function.
///
/// g(u) = ln(u)
impl LinkFunc for Log {
    fn func(x: f64) -> f64 {
        x.ln()
    }

    fn func_grad(x: f64) -> f64 {
        1f64 / x
    }

    fn func_inv(x: f64) -> f64 {
        x.exp()
    }
}

/// The Identity link function.
///
/// Used primarily as the canonical link in Linear Regression.
#[derive(Clone, Copy, Debug)]
pub struct Identity;

/// The Identity link function.
///
/// g(u) = u
impl LinkFunc for Identity {
    fn func(x: f64) -> f64 {
        x
    }

    fn func_grad(_: f64) -> f64 {
        1f64
    }

    fn func_inv(x: f64) -> f64 {
        x
    }
}

/// The Bernoulli regression family.
///
/// This is equivalent to logistic regression.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli;

impl Criterion for Bernoulli {
    type Link = Logit;

    fn model_variance(&self, mu: f64) -> f64 {
        let var = mu * (1f64 - mu);

        if var.abs() < 1e-10 {
            1e-10
        } else {
            var
        }
    }

    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let mut mu_data = Vec::with_capacity(y.len());

        for y_val in y {
            mu_data.push(if *y_val < 1e-10 {
                1e-10
            } else if *y_val > 1f64 - 1e-10 {
                1f64 - 1e-10
            } else {
                *y_val
            });
        }

        mu_data
    }

    fn compute_working_weight(&self, mu: &[f64]) -> Vec<f64> {
        let mut working_weights_vec = Vec::with_capacity(mu.len());

        for m in mu {
            let var = self.model_variance(*m);

            working_weights_vec.push(if var.abs() < 1e-5 {
                1e-5
            } else {
                var
            });
        }

        working_weights_vec
    }

    fn compute_y_bar(&self, y: &[f64], mu: &[f64]) -> Vec<f64> {
        let mut y_bar_vec = Vec::with_capacity(y.len());

        for (idx, m) in mu.iter().enumerate() {
            let target_diff = y[idx] - m;

            y_bar_vec.push(if target_diff.abs() < 1e-15 {
                0f64
            } else {
                self.link_grad(*m) * target_diff
            });
        }

        y_bar_vec
    }
}

/// The Binomial regression family.
#[derive(Debug)]
pub struct Binomial {
    weights: Vec<f64>,
}

impl Criterion for Binomial {
    type Link = Logit;

    fn model_variance(&self, mu: f64) -> f64 {
        let var = mu * (1f64 - mu);

        if var.abs() < 1e-10 {
            1e-10
        } else {
            var
        }

    }

    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let mut mu_data = Vec::with_capacity(y.len());

        for y_val in y {
            mu_data.push(if *y_val < 1e-10 {
                1e-10
            } else if *y_val > 1f64 - 1e-10 {
                1f64 - 1e-10
            } else {
                *y_val
            });
        }

        mu_data
    }

    fn compute_working_weight(&self, mu: &[f64]) -> Vec<f64> {
        let mut working_weights_vec = Vec::with_capacity(mu.len());

        for (idx, m) in mu.iter().enumerate() {
            let var = self.model_variance(*m) / self.weights[idx];

            working_weights_vec.push(if var.abs() < 1e-5 {
                1e-5
            } else {
                var
            });
        }

        working_weights_vec
    }

    fn compute_y_bar(&self, y: &[f64], mu: &[f64]) -> Vec<f64> {
        let mut y_bar_vec = Vec::with_capacity(y.len());

        for (idx, m) in mu.iter().enumerate() {
            let target_diff = y[idx] - m;

            y_bar_vec.push(if target_diff.abs() < 1e-15 {
                0f64
            } else {
                self.link_grad(*m) * target_diff
            });
        }

        y_bar_vec
    }
}

/// The Normal regression family.
///
/// This is equivalent to the Linear Regression model.
#[derive(Clone, Copy, Debug)]
pub struct Normal;

impl Criterion for Normal {
    type Link = Identity;

    fn model_variance(&self, _: f64) -> f64 {
        1f64
    }
}

/// The Poisson regression family.
#[derive(Clone, Copy, Debug)]
pub struct Poisson;

impl Criterion for Poisson {
    type Link = Log;

    fn model_variance(&self, mu: f64) -> f64 {
        mu
    }

    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let mut mu_data = Vec::with_capacity(y.len());

        for y_val in y {
            mu_data.push(if *y_val < 1e-10 {
                1e-10
            } else {
                *y_val
            });
        }

        mu_data
    }

    fn compute_working_weight(&self, mu: &[f64]) -> Vec<f64> {
        mu.to_vec()
    }
}
