//! Link Functions.
//!
//! This module contains a number of structs implementing the LinkFunc trait.
//!
//! These structs are used within Neural Networks and
//! Generalized Linear Regression (not yet implemented). 
//! 
//! You can also create your own custom Link Functions for use in your models.
//! Just create a unit struct implementing the LinkFunc trait.

/// Trait for link functions in models.
pub trait LinkFunc {
    fn func(x: f64) -> f64;

    fn func_grad(x: f64) -> f64;

    fn func_inv(x: f64) -> f64;
}

/// Sigmoid link function.
pub struct Sigmoid;

impl LinkFunc for Sigmoid {
    /// Sigmoid function.
    ///
    /// Returns 1 / ( 1 + e^-t).
    fn func(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Gradient of sigmoid function.
    ///
    /// Evaluates to (1 - e^-t) / (1 + e^-t)^2
    fn func_grad(x: f64) -> f64 {
        Self::func(x) * (1f64 - Self::func(x))
    }

    fn func_inv(x: f64) -> f64 {
        (x / (1f64-x)).ln()
    }
}

/// Linear link function.
pub struct Linear;

impl LinkFunc for Linear {
    fn func(x: f64) -> f64 {
        x
    }

    fn func_grad(_: f64) -> f64 {
        1f64
    }

    fn func_inv(x:f64) -> f64 {
        x
    }
}

/// Exponential link function.
pub struct Exp;

impl LinkFunc for Exp {
    fn func(x: f64) -> f64 {
        x.exp()
    }

    fn func_grad(x: f64) -> f64 {
        Self::func(x)
    }

    fn func_inv(x: f64) -> f64 {
        x.ln()
    }
}