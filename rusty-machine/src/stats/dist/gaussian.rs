//! Gaussian distribution module.
//!
//! Contains extension methods for the Normal struct
//! found in the rand crate. This is provided through
//! traits added within the containing stats module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::StandardNormal;
use super::consts as stat_consts;
use std::f64::consts as float_consts;

/// A Gaussian random variable.
///
/// This struct stores both the variance and the standard deviation.
/// This is to minimize the computation required for computing
/// the distribution functions and sampling.
///
/// It is most efficient to construct the struct using the `from_std_dev` constructor.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    mean: f64,
    variance: f64,
    _std_dev: f64,
}

/// The default Gaussian random variable.
/// This is the Standard Normal random variable.
///
/// The defaults are:
///
/// - mean = 0
/// - variance = 1
impl Default for Gaussian {
    fn default() -> Gaussian {
        Gaussian {
            mean: 0f64,
            variance: 1f64,
            _std_dev: 1f64,
        }
    }
}

impl Gaussian {
    /// Creates a new Gaussian random variable from
    /// a given mean and variance.
    pub fn new(mean: f64, variance: f64) -> Gaussian {
        Gaussian {
            mean: mean,
            variance: variance,
            _std_dev: variance.sqrt(),
        }
    }

    /// Creates a new Gaussian random variable from
    /// a given mean and standard deviation.
    pub fn from_std_dev(mean: f64, std_dev: f64) -> Gaussian {
        Gaussian {
            mean: mean,
            variance: std_dev * std_dev,
            _std_dev: std_dev,
        }
    }
}

/// The distribution of the gaussian random variable.
///
/// Accurately computes the PDF and log PDF.
/// Estimates the CDF accurate only to 0.003.
impl Distribution<f64> for Gaussian {
    /// The pdf of the normal distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Gaussian;
    /// use rusty_machine::stats::dist::Distribution;
    /// use rusty_machine::stats::dist::consts;
    ///
    /// let gauss = Gaussian::default();
    ///
    /// let lpdf_zero = gauss.pdf(0f64);
    ///
    /// // The value should be very close to 1/sqrt(2 * pi)
    /// assert!((lpdf_zero - (1f64/consts::SQRT_2_PI).abs()) < 1e-20);
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.mean) * (x - self.mean) / (2.0 * self.variance)).exp() /
        (stat_consts::SQRT_2_PI * self._std_dev)
    }

    /// The log pdf of the normal distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Gaussian;
    /// use rusty_machine::stats::dist::Distribution;
    /// use rusty_machine::stats::dist::consts;
    ///
    /// let gauss = Gaussian::default();
    ///
    /// let lpdf_zero = gauss.logpdf(0f64);
    ///
    /// // The value should be very close to -0.5*Ln(2 * pi)
    /// assert!((lpdf_zero + 0.5*consts::LN_2_PI).abs() < 1e-20);
    /// ```
    fn logpdf(&self, x: f64) -> f64 {
        -self._std_dev.ln() - (stat_consts::LN_2_PI / 2.0) -
        ((x - self.mean) * (x - self.mean) / (2.0 * self.variance))
    }

    /// Rough estimate for the cdf of the gaussian distribution.
    /// Accurate to 0.003.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Gaussian;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// let gauss = Gaussian::new(10f64, 5f64);
    /// let cdf_mid = gauss.cdf(10f64);
    ///
    /// assert!((0.5 - cdf_mid).abs() < 0.004);
    /// ```
    ///
    /// A slightly more involved test:
    ///
    /// ```
    /// use rusty_machine::stats::dist::Gaussian;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// let gauss = Gaussian::new(10f64, 4f64);
    /// let cdf = gauss.cdf(9f64);
    ///
    /// assert!((0.5*(1f64 - 0.382924922548) - cdf).abs() < 0.004);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        0.5 *
        (1f64 +
         (x - self.mean).signum() *
         (1f64 -
          (-float_consts::FRAC_2_PI * (x - self.mean) * (x - self.mean) / self.variance).exp())
            .sqrt())
    }
}

impl Sample<f64> for Gaussian {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.ind_sample(rng)
    }
}

impl IndependentSample<f64> for Gaussian {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let StandardNormal(n) = rng.gen::<StandardNormal>();
        self.mean + self._std_dev * n
    }
}
