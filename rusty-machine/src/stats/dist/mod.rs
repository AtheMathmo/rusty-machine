pub mod exponential;
pub mod gaussian;

pub use self::gaussian::Gaussian;
pub use self::exponential::Exponential;

/// Statistical constants
///
/// This module may be moved to the containing stats module in future.
pub mod consts {
    /// Sqrt(2 * pi)
    pub const SQRT_2_PI: f64 = 2.50662827463100050241576528481104525_f64;
    /// Ln(2 * pi)
    pub const LN_2_PI: f64 = 1.83787706640934548356065947281123527_f64;
}

/// Trait for statistical distributions.
pub trait Distribution<T> {
    /// The pdf of the distribution.
    fn pdf(&self, x: T) -> f64;

    /// The logpdf of the distribution.
    ///
    /// By default this takes the logarithm of the pdf.
    /// More efficient functions should be implemented.
    fn logpdf(&self, x: T) -> f64 {
        self.pdf(x).ln()
    }

    /// The cdf of the distribution.
    fn cdf(&self, x: T) -> f64;
}
