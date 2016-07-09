//! Logistic Regression module
//!
//! Contains implemention of logistic regression using
//! gradient descent optimization.
//!
//! The regressor will automatically add the intercept term
//! so you do not need to format the input matrices yourself.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::logistic_reg::LogisticRegressor;
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//!
//! let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
//! let targets = Vector::new(vec![0.,0.,1.,1.]);
//!
//! let mut log_mod = LogisticRegressor::default();
//!
//! // Train the model
//! log_mod.train(&inputs, &targets);
//!
//! // Now we'll predict a new point
//! let new_point = Matrix::new(1,1,vec![10.]);
//! let output = log_mod.predict(&new_point);
//!
//! // Hopefully we classified our new point correctly!
//! assert!(output[0] > 0.5, "Our classifier isn't very good!");
//! ```
//!
//! We could have been more specific about the learning of the model
//! by using the `new` constructor instead. This allows us to provide
//! a `GradientDesc` object with custom parameters.

use learning::SupModel;
use linalg::Matrix;
use linalg::Vector;
use learning::toolkit::activ_fn::{ActivationFunc, Sigmoid};
use learning::toolkit::cost_fn::{CostFunc, CrossEntropyError};
use learning::optim::grad_desc::GradientDesc;
use learning::optim::{OptimAlgorithm, Optimizable};

/// Logistic Regression Model.
///
/// Contains option for optimized parameter.
#[derive(Debug)]
pub struct LogisticRegressor<A>
    where A: OptimAlgorithm<BaseLogisticRegressor>
{
    base: BaseLogisticRegressor,
    alg: A,
}

/// Constructs a default Logistic Regression model
/// using standard gradient descent.
impl Default for LogisticRegressor<GradientDesc> {
    fn default() -> LogisticRegressor<GradientDesc> {
        LogisticRegressor {
            base: BaseLogisticRegressor::new(),
            alg: GradientDesc::default(),
        }
    }
}

impl<A: OptimAlgorithm<BaseLogisticRegressor>> LogisticRegressor<A> {
    /// Constructs untrained logistic regression model.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::logistic_reg::LogisticRegressor;
    /// use rusty_machine::learning::optim::grad_desc::GradientDesc;
    ///
    /// let gd = GradientDesc::default();
    /// let mut logistic_mod = LogisticRegressor::new(gd);
    /// ```
    pub fn new(alg: A) -> LogisticRegressor<A> {
        LogisticRegressor {
            base: BaseLogisticRegressor::new(),
            alg: alg,
        }
    }

    /// Get the parameters from the model.
    ///
    /// Returns an option that is None if the model has not been trained.
    pub fn parameters(&self) -> Option<&Vector<f64>> {
        self.base.parameters()
    }
}

impl<A> SupModel<Matrix<f64>, Vector<f64>> for LogisticRegressor<A>
    where A: OptimAlgorithm<BaseLogisticRegressor>
{
    /// Train the logistic regression model.
    ///
    /// Takes training data and output values as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::logistic_reg::LogisticRegressor;
    /// use rusty_machine::linalg::Matrix;
    /// use rusty_machine::linalg::Vector;
    /// use rusty_machine::learning::SupModel;
    ///
    /// let mut logistic_mod = LogisticRegressor::default();
    /// let inputs = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    /// let targets = Vector::new(vec![5.0, 6.0, 7.0]);
    ///
    /// logistic_mod.train(&inputs, &targets);
    /// ```
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        let initial_params = vec![0.5; full_inputs.cols()];

        let optimal_w = self.alg.optimize(&self.base, &initial_params[..], &full_inputs, targets);
        self.base.set_parameters(Vector::new(optimal_w));
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
        if let Some(v) = self.base.parameters() {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            (full_inputs * v).apply(&Sigmoid::func)
        } else {
            panic!("Model has not been trained.");
        }
    }
}

/// The Base Logistic Regression model.
///
/// This struct cannot be instantianated and is used internally only.
#[derive(Debug)]
pub struct BaseLogisticRegressor {
    parameters: Option<Vector<f64>>,
}

impl BaseLogisticRegressor {
    /// Construct a new BaseLogisticRegressor
    /// with parameters set to None.
    fn new() -> BaseLogisticRegressor {
        BaseLogisticRegressor { parameters: None }
    }
}

impl BaseLogisticRegressor {
    /// Returns a reference to the parameters.
    fn parameters(&self) -> Option<&Vector<f64>> {
        self.parameters.as_ref()
    }

    /// Set the parameters to `Some` vector.
    fn set_parameters(&mut self, params: Vector<f64>) {
        self.parameters = Some(params);
    }
}

/// Computing the gradient of the underlying Logistic
/// Regression model.
///
/// The gradient is given by
///
/// X<sup>T</sup>(h(Xb) - y) / m
///
/// where `h` is the sigmoid function and `b` the underlying model parameters.
impl Optimizable for BaseLogisticRegressor {
    type Inputs = Matrix<f64>;
    type Targets = Vector<f64>;

    fn compute_grad(&self,
                    params: &[f64],
                    inputs: &Matrix<f64>,
                    targets: &Vector<f64>)
                    -> (f64, Vec<f64>) {

        let beta_vec = Vector::new(params.to_vec());
        let outputs = (inputs * beta_vec).apply(&Sigmoid::func);

        let cost = CrossEntropyError::cost(&outputs, targets);
        let grad = (inputs.transpose() * (outputs - targets)) / (inputs.rows() as f64);

        (cost, grad.into_vec())
    }
}
