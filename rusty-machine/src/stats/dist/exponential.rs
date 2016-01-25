//! Exponential distribution module.
//!
//! Contains extension methods for the Exp struct
//! found in the rand crate. This is provided through
//! traits added within the containing stats module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::exponential::Exp1;

pub struct Exponential {
    lambda: f64,
}

impl Distribution<f64> for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        assert!(x >= 0., "Input to pdf must be positive for exponential.");
        (-x * self.lambda).exp() * self.lambda
    }

    fn logpdf(&self, x: f64) -> f64 {
        assert!(x >= 0., "Input to log pdf must be positive for exponential.");
        self.lambda.ln() - (x * self.lambda)
    }

    fn cdf(&self, x: f64) -> f64 {
        assert!(x >= 0., "Input to cdf must be positive for exponential.");
        1.0 - (-x * self.lambda).exp()
    }
}

impl Sample<f64> for Exponential {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}

impl IndependentSample<f64> for Exponential {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let Exp1(n) = rng.gen::<Exp1>();
        n / self.lambda
    }
}
