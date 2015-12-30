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

/// Linear Regression Model.
///
/// Contains option for optimized parameter.
pub struct LinRegressor {
    pub b: Option<Vector<f64>>,
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
    /// let data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    /// let values = Vector::new(vec![5.0, 6.0, 7.0]);
    ///
    /// lin_mod.train(&data, &values);
    /// ```
    fn train(&mut self, data: &Matrix<f64>, values: &Vector<f64>) {
        let xt = data.transpose();

        self.b = Some(((&xt * data).inverse() * &xt) * values);
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, data: &Matrix<f64>) -> Vector<f64> {
        match self.b {
            Some(ref v) => data * v,
            None => panic!("Model has not been trained."),
        }
    }
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
        LinRegressor { b: None }
    }
}