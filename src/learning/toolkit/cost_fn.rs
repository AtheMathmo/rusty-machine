//! Cost Functions.
//!
//! This module contains a number of structs implementing the `CostFunc` trait.
//!
//! These structs are used within Neural Networks and
//! Generalized Linear Regression (not yet implemented).
//!
//! You can also create your own custom cost functions for use in your models.
//! Just create a struct implementing the `CostFunc` trait.

use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use linalg::Vector;

/// Trait for cost functions in models.
pub trait CostFunc<T> {
    /// The cost function.
    fn cost(outputs: &T, targets: &T) -> f64;

    /// The gradient of the cost function.
    fn grad_cost(outputs: &T, targets: &T) -> T;
}

/// The mean squared error cost function.
#[derive(Clone, Copy, Debug)]
pub struct MeanSqError;

// For generics we need a trait for "Hadamard product" here
// Which is "Elementwise multiplication".
impl CostFunc<Matrix<f64>> for MeanSqError {
    fn cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
        let diff = outputs - targets;
        let sq_diff = &diff.elemul(&diff);

        let n = diff.rows();

        sq_diff.sum() / (2f64 * (n as f64))
    }

    fn grad_cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> Matrix<f64> {
        outputs - targets
    }
}

impl CostFunc<Vector<f64>> for MeanSqError {
    fn cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> f64 {
        let diff = outputs - targets;
        let sq_diff = &diff.elemul(&diff);

        let n = diff.size();

        sq_diff.sum() / (2f64 * (n as f64))
    }

    fn grad_cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> Vector<f64> {
        outputs - targets
    }
}

/// The cross entropy error cost function.
#[derive(Clone, Copy, Debug)]
pub struct CrossEntropyError;

impl CostFunc<Matrix<f64>> for CrossEntropyError {
    fn cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
        // The cost for a single
        let log_inv_output = (-outputs + 1f64).apply(&ln);
        let log_output = outputs.clone().apply(&ln);

        let mat_cost = targets.elemul(&log_output) + (-targets + 1f64).elemul(&log_inv_output);

        let n = outputs.rows();

        -(mat_cost.sum()) / (n as f64)
    }

    fn grad_cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> Matrix<f64> {
        (outputs - targets).elediv(&(outputs.elemul(&(-outputs + 1f64))))
    }
}

impl CostFunc<Vector<f64>> for CrossEntropyError {
    fn cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> f64 {
        // The cost for a single
        let log_inv_output = (-outputs + 1f64).apply(&ln);
        let log_output = outputs.clone().apply(&ln);

        let mat_cost = targets.elemul(&log_output) + (-targets + 1f64).elemul(&log_inv_output);

        let n = outputs.size();

        -(mat_cost.sum()) / (n as f64)
    }

    fn grad_cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> Vector<f64> {
        (outputs - targets).elediv(&(outputs.elemul(&(-outputs + 1f64))))
    }
}

/// Logarithm for applying within cost function.
fn ln(x: f64) -> f64 {
    x.ln()
}
