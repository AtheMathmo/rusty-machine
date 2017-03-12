//! Poiisson distribution module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use super::utils::{lgamma};

/// Aa Poisson random variable.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    lambda: f64,
}

/// The default Poisson random variable.
///
/// The defaults are:
///
/// - lambda = 1
impl Default for Poisson {
    fn default() -> Poisson {
        Poisson { lambda: 1f64 }
    }
}

impl Poisson {
    /// Constructs a new Poisson random variable with given
    /// lambda parameter.
    pub fn new(lambda: f64) -> Poisson {
        Poisson { lambda: lambda }
    }

    /// Returns the lambda parameter.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution<u64> for Poisson {
    /// The pdf of the poisson distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Poisson;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct an poisson with lambda parameter 5.0.
    /// let poisson = Poisson::new(5f64);
    ///
    /// let pdf_zero = poisson.pdf(0);
    /// assert!((pdf_zero - (-poisson.lambda()).exp()).abs() < 1e-20);
    /// ```
    fn pdf(&self, k: u64) -> f64 {
        self.logpdf(k).exp()
    }

    /// The log pdf of the poisson distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// // Construct an poisson with lambda parameter 5.0.
    /// use rusty_machine::stats::dist::Poisson;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct an poisson with lambda parameter 5.0.
    /// let poisson = Poisson::new(7f64);
    ///
    /// let log_pdf_0 = poisson.logpdf(0);
    ///
    /// assert!((log_pdf_0 - (-poisson.lambda())).abs() < 1e-20);
    /// ```
    fn logpdf(&self, k: u64) -> f64 {
        let fk = k as f64;
        fk as f64 * self.lambda.ln() - self.lambda - lgamma(fk + 1.0)
    }

    /// The cdf of the poisson distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Poisson;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a poisson with lambda parameter 5.0.
    /// let poisson = Poisson::new(5f64);
    ///
    /// let cdf_zero = poisson.cdf(0);
    ///
    /// assert!((cdf_zero).abs() < 1e-20);
    /// ```
    fn cdf(&self, k: u64) -> f64 {
        return (0..k).map(|i| self.pdf(i)).sum();

    }
}

/// Samples from the poisson distribution for small values of lambda.
/// This uses Knuth's algorithm.
fn poisson_knuth<R: Rng>(lambda: f64, r: &mut R) -> u64 {

    let lamexp = (-lambda).exp();
    let mut x = 0;
    let mut result = 1.0;
    loop
    {
        let u = r.next_f64();
        result *= u;
        if result > lamexp {
            x += 1;
        }
        else {
            return x;
        }
    }
}

/// Samples from the poisson distributions for large values of lambda.
///
/// It uses the transformed rejection method for generating Poisson random variables
/// [Hoerman](http://dx.doi.org/10.1016/0167-6687(93)90997-4).  The code is informed by
/// [NumPy's implementation](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/distributions.c)
fn poisson_hoermann<R: Rng>(lambda: f64, r: &mut R) -> u64 {

    let loglam = lambda.ln();
    let b = 0.931 + 2.53 * lambda.sqrt();
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328/(b-3.4);
    let vr = 0.9277 - 3.6224/(b-2.0);

    loop
    {
        let u = r.next_f64() - 0.5;
        let v = r.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0*a/us + b)*u + lambda + 0.43).trunc() as i64;
        if (us >= 0.07) && (v <= vr) {
            return k as u64;
        }
        if (k < 0) || ((us < 0.013) && (v > us)) {
            continue;
        }
        if (v.ln() + invalpha.ln() - (a/(us*us)+b).ln()) <=
            (-lambda + k as f64*loglam - lgamma(k as f64+1.0)) {
            return k as u64;
        }
    }

}



impl Sample<u64> for Poisson {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> u64 {
        self.ind_sample(rng)
    }

}

impl IndependentSample<u64> for Poisson {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> u64 {
        if self.lambda >= 10.0 {
            return poisson_hoermann(self.lambda, rng);
        }
        else if self.lambda == 0.0 {
            return 0;
        }
        else {
            return poisson_knuth(self.lambda, rng);
        }

    }
}
