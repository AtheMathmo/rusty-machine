//! Linear Regression module
//!
//! Contains implemention of linear regression using
//! OLS and gradient descent optimization.
//!
//! The regressor will automatically add the intercept term
//! so you do not need to format the input matrices yourself.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::lin_reg::LinRegressor;
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//!
//! let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
//! let targets = Vector::new(vec![1.,5.,9.,13.]);
//!
//! let mut lin_mod = LinRegressor::default();
//!
//! // Train the model
//! lin_mod.train(&inputs, &targets).unwrap();
//!
//! // Now we'll predict a new point
//! let new_point = Matrix::new(1,1,vec![10.]);
//! let output = lin_mod.predict(&new_point).unwrap();
//!
//! // Hopefully we classified our new point correctly!
//! assert!(output[0] > 17f64, "Our regressor isn't very good!");
//! ```

use learning::error::Error;
use learning::optim::grad_desc::GradientDesc;
use learning::optim::{OptimAlgorithm, Optimizable};
use learning::toolkit::cost_fn::CostFunc;
use learning::toolkit::cost_fn::MeanSqError;
use learning::{LearningResult, SupModel};
use linalg::Vector;
use linalg::{BaseMatrix, Matrix};

/// Linear Regression Model.
///
/// Contains option for optimized parameter.
#[derive(Debug)]
pub struct LinRegressor {
    /// The parameters for the regression model.
    parameters: Option<Vector<f64>>,
}

impl Default for LinRegressor {
    fn default() -> LinRegressor {
        LinRegressor { parameters: None }
    }
}

impl LinRegressor {
    /// Get the parameters from the model.
    ///
    /// Returns an option that is None if the model has not been trained.
    pub fn parameters(&self) -> Option<&Vector<f64>> {
        self.parameters.as_ref()
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
    /// use rusty_machine::linalg::Matrix;
    /// use rusty_machine::linalg::Vector;
    /// use rusty_machine::learning::SupModel;
    ///
    /// let mut lin_mod = LinRegressor::default();
    /// let inputs = Matrix::new(3,1, vec![2.0, 3.0, 4.0]);
    /// let targets = Vector::new(vec![5.0, 6.0, 7.0]);
    ///
    /// lin_mod.train(&inputs, &targets).unwrap();
    /// ```
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) -> LearningResult<()> {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        let xt = full_inputs.transpose();
        self.parameters = Some((&xt * full_inputs).solve(&xt * targets)?);
        Ok(())
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<f64>> {
        if let Some(ref v) = self.parameters {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            Ok(full_inputs * v)
        } else {
            Err(Error::new_untrained())
        }
    }
}

impl Optimizable for LinRegressor {
    type Inputs = Matrix<f64>;
    type Targets = Vector<f64>;

    fn compute_grad(
        &self,
        params: &[f64],
        inputs: &Matrix<f64>,
        targets: &Vector<f64>,
    ) -> (f64, Vec<f64>) {
        let beta_vec = Vector::new(params.to_vec());
        let outputs = inputs * beta_vec;

        let cost = MeanSqError::cost(&outputs, targets);
        let grad = (inputs.transpose() * (outputs - targets)) / (inputs.rows() as f64);

        (cost, grad.into_vec())
    }
}

impl LinRegressor {
    /// Train the linear regressor using Gradient Descent.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::lin_reg::LinRegressor;
    /// use rusty_machine::learning::SupModel;
    /// use rusty_machine::linalg::Matrix;
    /// use rusty_machine::linalg::Vector;
    ///
    /// let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
    /// let targets = Vector::new(vec![1.,5.,9.,13.]);
    ///
    /// let mut lin_mod = LinRegressor::default();
    ///
    /// // Train the model
    /// lin_mod.train_with_optimization(&inputs, &targets);
    ///
    /// // Now we'll predict a new point
    /// let new_point = Matrix::new(1,1,vec![10.]);
    /// let _ = lin_mod.predict(&new_point).unwrap();
    /// ```
    pub fn train_with_optimization(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        let initial_params = vec![0.; full_inputs.cols()];

        let gd = GradientDesc::default();
        let optimal_w = gd.optimize(self, &initial_params[..], &full_inputs, targets);
        self.parameters = Some(Vector::new(optimal_w));
    }
}
