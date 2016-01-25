//! Module for kernels
//!
//! Currently used within Gaussian Processes.

use linalg::vector::Vector;

/// The Kernel trait
///
/// Requires a function mapping two vectors to a scalar.
pub trait Kernel {
    /// The kernel function.
    ///
    /// Takes two equal length slices and returns a scalar.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64;
}

/// Squared exponential kernel
///
/// Equivalently a gaussian function.
///
/// The kernel function is given by:
///
/// f(x1,x2) = A _exp_(-||x1-x2||^2 / 2(l^2))
///
/// Where A is the amplitude and l the length scale.
pub struct SquaredExp {
    ls: f64,
    ampl: f64,
}

impl SquaredExp {
    /// Construct a new squared exponential kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel::SquaredExp;
    ///
    /// let ker = SquaredExp::new(2f64, 1f64);
    /// ```
    pub fn new(ls: f64, ampl: f64) -> SquaredExp {
        SquaredExp {
            ls: ls,
            ampl: ampl,
        }
    }
}

impl Default for SquaredExp {
    /// Constructs the default Squared Exp kernel.
    ///
    /// The default settings are:
    /// - length scale = 1
    /// - amplitude = 1
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel::SquaredExp;
    ///
    /// let ker = SquaredExp::default();
    /// ```
    fn default() -> SquaredExp {
        SquaredExp {
            ls: 1f64,
            ampl: 1f64,
        }
    }
}

impl Kernel for SquaredExp {
    /// The squared exponential kernel function.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());

        let v1 = Vector::new(x1.to_vec());
        let v2 = Vector::new(x2.to_vec());

        let x = -(&v1 - &v2).dot(&(v1 - v2)) / (2f64 * self.ls * self.ls);
        (self.ampl * x.exp())
    }
}
