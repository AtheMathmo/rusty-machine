//! Exponential distribution module.
//!
//! Contains extension methods for the Exp struct
//! found in the rand crate. This is provided through
//! traits added within the containing stats module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::exponential::Exp1;

/// An Exponential random variable.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    lambda: f64,
}

/// The default Exponential random variable.
///
/// The defaults are:
///
/// - lambda = 1
impl Default for Exponential {
    fn default() -> Exponential {
        Exponential { lambda: 1f64 }
    }
}

impl Exponential {
    /// Constructs a new Exponential random variable with given
    /// lambda parameter.
    pub fn new(lambda: f64) -> Exponential {
        Exponential { lambda: lambda }
    }

    /// Returns the lambda parameter.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution<f64> for Exponential {
    /// The pdf of the exponential distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Exponential;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct an exponential with lambda parameter 7.0.
    /// let exp = Exponential::new(7f64);
    ///
    /// let pdf_zero = exp.pdf(0f64);
    /// assert!((pdf_zero - exp.lambda()).abs() < 1e-20);
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        assert!(x >= 0., "Input to pdf must be positive for exponential.");
        (-x * self.lambda).exp() * self.lambda
    }

    /// The log pdf of the exponential distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// // Construct an exponential with lambda parameter 5.0.
    /// use rusty_machine::stats::dist::Exponential;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct an exponential with lambda parameter 5.0.
    /// let exp = Exponential::new(5f64);
    ///
    /// let log_pdf = exp.logpdf(3f64);
    ///
    /// assert!((log_pdf - exp.lambda().ln() + exp.lambda() * 3f64).abs() < 1e-20);
    /// ```
    fn logpdf(&self, x: f64) -> f64 {
        assert!(x >= 0.,
                "Input to log pdf must be positive for exponential.");
        self.lambda.ln() - (x * self.lambda)
    }

    /// The cdf of the exponential distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Exponential;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct an exponential with lambda parameter 5.0.
    /// let exp = Exponential::new(5f64);
    ///
    /// let cdf_zero = exp.cdf(0f64);
    ///
    /// assert!((cdf_zero).abs() < 1e-20);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        assert!(x >= 0., "Input to cdf must be positive for exponential.");
        1.0 - (-x * self.lambda).exp()
    }
}

impl Sample<f64> for Exponential {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.ind_sample(rng)
    }
}

impl IndependentSample<f64> for Exponential {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let Exp1(n) = rng.gen::<Exp1>();
        n / self.lambda
    }
}
