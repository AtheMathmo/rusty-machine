//! Activation Functions.
//!
//! This module contains a number of structs implementing the `ActivationFunc` trait.
//!
//! These structs are used within Neural Networks and
//! Generalized Linear Regression (not yet implemented).
//!
//! You can also create your own custom activation Functions for use in your models.
//! Just create a unit struct implementing the `ActivationFunc` trait.

use std::fmt::Debug;

/// Trait for activation functions in models.
pub trait ActivationFunc: Clone + Debug {
    /// The activation function.
    fn func(x: f64) -> f64;

    /// The gradient of the activation function.
    fn func_grad(x: f64) -> f64;

    /// The gradient of the activation function calculated using the output of the function.
    /// Calculates f'(x) given f(x) as an input
    fn func_grad_from_output(y: f64) -> f64;

    /// The inverse of the activation function.
    fn func_inv(x: f64) -> f64;
}

/// Sigmoid activation function.
#[derive(Clone, Copy, Debug)]
pub struct Sigmoid;

impl ActivationFunc for Sigmoid {
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

    fn func_grad_from_output(y: f64) -> f64 {
        y * (1f64 - y)
    }

    fn func_inv(x: f64) -> f64 {
        (x / (1f64 - x)).ln()
    }
}

/// Linear activation function.
#[derive(Clone, Copy, Debug)]
pub struct Linear;

impl ActivationFunc for Linear {
    fn func(x: f64) -> f64 {
        x
    }

    fn func_grad(_: f64) -> f64 {
        1f64
    }

    fn func_grad_from_output(_: f64) -> f64 {
        1f64
    }

    fn func_inv(x: f64) -> f64 {
        x
    }
}

/// Exponential activation function.
#[derive(Clone, Copy, Debug)]
pub struct Exp;

impl ActivationFunc for Exp {
    fn func(x: f64) -> f64 {
        x.exp()
    }

    fn func_grad(x: f64) -> f64 {
        Self::func(x)
    }

    fn func_grad_from_output(y: f64) -> f64 {
        y
    }

    fn func_inv(x: f64) -> f64 {
        x.ln()
    }
}

/// Hyperbolic tangent activation function
#[derive(Clone, Copy, Debug)]
pub struct Tanh;

impl ActivationFunc for Tanh {
    fn func(x: f64) -> f64 {
        x.tanh()
    }

    fn func_grad(x: f64) -> f64 {
        let y = x.tanh();
        1.0 - y*y
    }

    fn func_grad_from_output(y: f64) -> f64 {
        1.0 - y*y
    }

    fn func_inv(x: f64) -> f64 {
        0.5*((1.0+x)/(1.0-x)).ln()
    }
}