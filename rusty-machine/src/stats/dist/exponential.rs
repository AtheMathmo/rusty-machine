//! Exponential distribution module.
//!
//! Contains 

use stats::dist::Distribution;

/// Exponential distribution struct.
pub struct Exponential {
	pub lambda: f64,
}

impl Exponential {

	/// Constructs a new Exponential distribution.
	///
	/// # Examples
	///
	/// ```
	/// use rusty_machine::stats::dist::exponential::Exponential;
	///
	/// let x = Exponential::new(2.0);
	/// ```
	pub fn new(lambda: f64) -> Exponential {
		assert!(lambda > 0.);
		Exponential { lambda: lambda }
	}
}
impl Distribution<f64> for Exponential {
	fn pdf(&self, x:f64) -> f64 {
		assert!(x >= 0.);
		self.lambda * (-self.lambda * x).exp()
	}

	fn logpdf(&self, x:f64) -> f64 {
		assert!(x >= 0.);
		self.lambda.ln() - self.lambda * x
	}

	fn cdf(&self, x:f64) -> f64 {
		assert!(x >= 0.);
		1.0 - (-self.lambda * x).exp()
	}
}