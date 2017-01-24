//! Binomial distribution module.
//!
//! Contains extension methods for the Binomial struct
//! found in the rand crate. This is provided through
//! traits added within the containing stats module.

use stats::dist::Distribution;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};

use std::cmp::min;


/// A Binomial random variable.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    experiments: u64,
    p: f64
}

/// The default Binomial random variable.
///
/// The defaults are:
///
/// - experiments: 1
/// - p: 0.5
impl Default for Binomial {
    fn default() -> Binomial {
        Binomial { experiments: 1, p: 0.5 }
    }
}

impl Binomial {
    /// Constructs a new Binonial random variable with given
    /// numer of experiments and p values.
    pub fn new(experiments: u64, p:f64) -> Binomial {
        Binomial { experiments: experiments, p: p }
    }

}

/// From n objects choose p
// Defined as n!/p!(n - p)!
fn choose(n: u64, p:u64) -> u64 {
    assert!(p <= n, "p must be less than n in choose function");
    let mut q = n - p;
    let mut pmut = p;
    if q < p {
        // swap p and q
        pmut += q;
        q = pmut - q;
        pmut -= q;
    }
    let mut top = 1;
    let mut bottom = 1;
    for i in 2..pmut + 1 {
        bottom *= i
    }
    for i in q + 1..n + 1 {
        top *= i
    }
    return top / bottom;
}

impl Distribution<u64> for Binomial {
    /// The pdf of the binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Binomial;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a binomial distribution for two coin flips.
    /// let bin = Binomial::new(2, 0.5);
    ///
    /// let pdf_one = bin.pdf(1);
    /// assert_eq!(pdf_one, 0.5);
    /// ```
    fn pdf(&self, x: u64) -> f64 {
        if x > self.experiments {
            return 0.0
        }
        choose(self.experiments, x) as f64 * self.p.powi(x as i32) * (1.0 - self.p).powi(self.experiments as i32 - x as i32)
    }

    /// The log pdf of the binomial distribution.
    ///
    /// Implemented simply as the result of calling the log on the pdf function
    /// ```
    fn logpdf(&self, x: u64) -> f64 {
        // Theres not really a simpler form than just taking the ln of the result.
        // For large N, the whole thing can be approximated by the normal distribution, but
        // that won't really help us here.  Further reading:
        // http://mathworld.wolfram.com/BinomialDistribution.html
        return self.pdf(x).ln()
    }

    /// The cdf of the multinomial distribution.
    ///
    /// Implemented simply as the sum of all pdfs lower than it.  There is no closed form
    /// solution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Binomial;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a binomial with 200 trials with probability of success 1/4.
    /// let bin = Binomial::new(20, 0.25);
    ///
    /// let cdf_five = bin.cdf(5);
    ///
    /// assert!(cdf_five - 0.6171726543871046 < 1e-20);
    /// ```
    fn cdf(&self, x: u64) -> f64 {
        let mut total = 0.0;
        for i in 0..x + 1 {
            total += self.pdf(i);
        }
        total
    }
}

impl Sample<u64> for Binomial {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> u64 {
        self.ind_sample(rng)
    }
}

#[derive(Copy, Clone)]
enum Step {
    Step1,
    Step2,
    Step3,
    Step4,
    Step5,
    Step6,
    Step7
}


/// This function samples from a binomial distribution with n trials and a p probability of success
/// The algorithm is based on
/// [Kachitvichyanukul and Schmeiser](https://doi-org.ezproxy.library.dal.ca/10.1145/42372.42381)
/// as implemented by
/// [randomkit in numpy](https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/distributions.c).
/// As to what is actually going on in this algorithm, good luck.
fn binomial_btpe<R: Rng>(n: u64, p: f64, rng: &mut R) -> u64 {
    let r = p;
    let q = 1.0 - r;
    let fm = n as f64 * r + r;
    let m = fm as u64 as f64;
    let nrq = n as f64 * r * q;
    let p1 = (2.195 * nrq.sqrt() -4.6 * q) as u64 as f64 + 0.5;
    let xm = m + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5/(15.3 + m);
    let a = (fm - xl)/(fm -xl*r);
    let laml = a*(1.0+ a/2.0);
    let a = (xr - fm)/ (xr * q);
    let lamr = a * (1.0 + a/ 2.0);
    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / laml;
    let p4 = p3 + c / lamr;
    let mut u = 0.0;
    let mut v = 0.0;
    let mut y = 0;
    let mut k = 0.0;

    let mut step = Step::Step1;
    loop {
        step = match step {
            Step::Step1 => {
                u = rng.next_f64() * p4;
                v = rng.next_f64();
                if u > p1 {
                    Step::Step2
                } else {
                    y = (xm - p1 * v + u) as i64;
                    Step::Step7
                }
            }, Step::Step2 => {
                if u > p3 {
                    Step::Step3
                } else {
                    let x = xl + (u - p1) / c;
                    v = v * c + 1.0 - (m - x + 0.5).abs() / p1;
                    if v > 1.0 {
                        Step::Step1
                    } else {
                        y = x as i64;
                        Step::Step5
                    }
                }
            }, Step::Step3 => {
                if u > p3 {
                    Step::Step4
                } else {
                    y = (xl + v.ln() / laml) as i64;
                    if y < 0 {
                        Step::Step1
                    } else {
                        v = v*(u-p2)*laml;
                        Step::Step5
                    }
                }
            }, Step::Step4 => {
                y = (xr - v.ln() / lamr) as i64;
                if y > n as i64 {
                    Step::Step1
                } else {
                    v = v * (u - p3) * lamr;
                    Step::Step5
                }
            }, Step::Step5 => {
                k = (y as f64 - m).abs();
                if k > 20.0 && k < nrq/2.0 - 1.0 {
                    Step::Step6
                } else {
                    let s = r / q;
                    let a = s * (n as f64 + 1.0);
                    let mut f = 1.0;
                    if m < y as f64 {
                        for i in m as u64 + 1..y as u64 + 1 {
                            f *= a/i as f64 - s;
                        }
                    } else if m as i64 > y {
                        for i in y as u64 + 1..m as u64 + 1 {
                            f *= a/i as f64 - s;
                        }
                    }
                    if v > f {
                        Step::Step1
                    } else {
                        Step::Step7
                    }
                }
            }, Step::Step6 => {
                let rho = (k/(nrq))*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5);
                let t = -k*k/(2.0*nrq);
                let a = v.ln();
                if a < (t - rho) {
                    Step::Step7
                } else if a > (t + rho) {
                    Step::Step1
                } else {
                    let x1 = y as f64 +1.0;
                    let f1 = m + 1.0;
                    let z = n as f64 +1.0-m ;
                    let w = (n as i64-y+1) as f64;
                    let x2 = x1*x1;
                    let f2 = f1*f1;
                    let z2 = z*z;
                    let w2 = w*w;
                    if a > (xm*(f1/x1).ln()
                           + (n as f64-m+0.5)*(z/w).ln()
                           + (y as f64-m)*(w*r/(x1*q)).ln()
                           + (13680.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
                           + (13680.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
                           + (13680.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
                           + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.) {
                        Step::Step1
                    } else {
                        Step::Step7
                    }
                }
            }, Step::Step7 =>  {
                return y as u64
            }

        };
    }
}

fn binomial_inversion<R: Rng>(n: u64, p: f64, rng: &mut R) -> u64 {
    let q = 1.0 - p;
    let qn = (n as f64 * q.ln()).exp();
    let np = n as f64 * p;
    let bound:u64 = min(n, (np + 10.0 * (np * q + 1.0).sqrt()) as u64);
    let mut x:u64 = 0;
    let mut px = qn;
    let mut u = rng.next_f64();
    while u > px {
        x += 1;
        if x > bound {
            x = 0;
            px = qn;
            u = rng.next_f64();
        } else {
            u -= px;
            px = ((n - x + 1) as f64 * p * px) / (x as f64 * q);
        }
    }
    x
}

impl IndependentSample<u64> for Binomial {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> u64 {
        if self.p <= 0.5 {
            if self.p * self.experiments as f64 <= 30.0 {
                binomial_inversion(self.experiments, self.p, rng)
            } else {
                binomial_btpe(self.experiments, self.p, rng)
            }
        } else {
            let q = 1.0 - self.p;
            if q * self.experiments as f64 <= 30.0 {
                self.experiments - binomial_inversion(self.experiments, q, rng)
            } else {
                self.experiments - binomial_btpe(self.experiments, q, rng)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::choose;
    #[test]
    fn test_choose() {
        assert_eq!(choose(4, 2), 6);
        assert_eq!(choose(10, 7), 120);
        assert_eq!(choose(9, 0), 1);
        assert_eq!(choose(100, 100), 1);
        assert_eq!(choose(25, 17), 1081575);
    }
}
