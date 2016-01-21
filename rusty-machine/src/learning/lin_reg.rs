//! Linear Regression module
//!
//! Contains implementation of linear regression models.
//! Allows training and prediction of linear regression model
//! using least squares optimization.
//!
//! Currently only OLS solution - gradient descent not yet implemented.

use learning::SupModel;
use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::toolkit::cost_fn::CostFunc;
use learning::toolkit::cost_fn::MeanSqError;
use learning::optim::grad_desc::GradientDesc;
use learning::optim::OptimAlgorithm;
use learning::optim::Optimizable;

/// Linear Regression Model.
///
/// Contains option for optimized parameter.
pub struct LinRegressor {
    /// The parameters for the regression model.
    parameters: Option<Vector<f64>>,
}

impl LinRegressor {
    /// Constructs untrained linear regression model.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::lin_reg::LinRegressor;
    ///
    /// let mut lin_mod = LinRegressor::new();
    /// ```
    pub fn new() -> LinRegressor {
        LinRegressor { parameters: None }
    }

    /// Get the parameters from the model.
    ///
    /// Returns an option that is None if the model has not been trained.
    pub fn parameters(&self) -> Option<Vector<f64>> {
        match self.parameters {
            None => None,
            Some(ref x) => Some(x.clone()),
        }
    }
}

impl SupModel<Matrix<f64>, Vector<f64>> for LinRegressor {
    /// Train the linear regression model.
    ///
    /// Takes training data and output values as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::lin_reg::LinRegressor;
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::vector::Vector;
    /// use rusty_machine::learning::SupModel;
    ///
    /// let mut lin_mod = LinRegressor::new();
    /// let inputs = Matrix::new(3,1, vec![2.0, 3.0, 4.0]);
    /// let targets = Vector::new(vec![5.0, 6.0, 7.0]);
    ///
    /// lin_mod.train(&inputs, &targets);
    /// ```
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        let xt = full_inputs.transpose();

        self.parameters = Some(((&xt * full_inputs).inverse() * &xt) * targets);
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
        if let Some(ref v) = self.parameters {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            full_inputs * v
        }
        else {
            panic!("Model has not been trained.");
        }
    }
}

impl Optimizable for LinRegressor {
    type Inputs = Matrix<f64>;
    type Targets = Vector<f64>;

    fn compute_grad(&self, params: &[f64], inputs: &Matrix<f64>, targets: &Vector<f64>) -> (f64, Vec<f64>) {
        
        let beta_vec = Vector::new(params.to_vec());
        let outputs = inputs * beta_vec;

        let cost = MeanSqError::cost(&outputs, targets);
        let grad = (inputs.transpose() * (outputs-targets)) / (inputs.rows() as f64);

        (cost, grad.into_vec())
    }
}

impl LinRegressor {
    pub fn train_with_optimization(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        let initial_params = vec![0.; full_inputs.cols()];

        let gd = GradientDesc::default();
        let optimal_w = gd.optimize(self, &initial_params[..], &full_inputs, targets);
        self.parameters = Some(Vector::new(optimal_w));
    }
}