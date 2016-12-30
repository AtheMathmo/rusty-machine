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

use linalg::Vector;

mod lin_reg_impl;
mod ridge_reg_impl;

/// Linear Regression Model.
///
/// Contains option for optimized parameter.
#[derive(Debug)]
pub struct LinRegressor {
    /// The parameters for the regression model.
    parameters: Option<Vector<f64>>,
}

/// Ridge Regression Model.
///
/// Contains option for optimized parameter.
#[derive(Debug)]
pub struct RidgeRegressor {
    alpha: f64,
    /// The parameters for the regression model.
    parameters: Option<Vector<f64>>,
}