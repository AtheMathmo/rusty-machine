//! Gaussian distribution module.
//!
//! Contains extension methods for the Normal struct
//! found in the rand crate. This is provided through
//! traits added within the containing stats module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::StandardNormal;
use super::consts;

pub struct Gaussian {
    mean: f64,
    variance: f64,
    _std_dev: f64,
}

impl Default for Gaussian {
    fn default() -> Gaussian {
        Gaussian(0f64, 1f64, 1f64)
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

impl Distribution<f64> for Gaussian {
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.mean) * (x - self.mean) / (2.0 * self.variance)).exp() /
        (consts::SQRT_2_PI * self._std_dev)
    }

    fn logpdf(&self, x: f64) -> f64 {
        -self._std_dev.log() - (consts::LN_2_PI/2.0) -
        ((x - self.mean) * (x - self.mean) / (2.0 * self.variance))
    }

    fn cdf(&self, x: f64) -> f64 {
        unimplemented!();
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
