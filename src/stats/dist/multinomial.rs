//! Multinomial distribution module.

use stats::dist::Distribution;
use super::binomial::Binomial;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use std::cmp::{min, max};


/// A Multinomial random variable.
#[derive(Debug, Clone)]
pub struct Multinomial {
    experiments: u32,
    p_vals: Vec<f64>
}

/// The default Multinomial random variable.
///
/// The defaults are:
///
/// - experiments: 1
/// - p_vals: vec![0.5, 0.5]
impl Default for Multinomial {
    fn default() -> Multinomial {
        Multinomial { experiments: 1, p_vals: vec![0.5, 0.5] }
    }
}

impl Multinomial {
    /// Constructs a new Multinomial random variable with given
    /// numer of experiments and p values.
    pub fn new(experiments: u32, p_vals:Vec<f64>) -> Multinomial {
        assert!(p_vals.len() > 0, "Multinomial p_vals must have length greater than 0");
        Multinomial { experiments: experiments, p_vals: p_vals }
    }

}

impl Distribution<Vec<u32> > for Multinomial {
    /// The pdf of the multinomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Multinomial;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a multinomial distribution for two four sided dice roles.
    /// let multi = Multinomial::new(2, vec![0.25, 0.25, 0.25, 0.25]);
    ///
    /// let pdf_one = multi.pdf(vec![0, 1, 1, 0]);
    /// assert_eq!(pdf_one, 0.125);
    /// ```
    ///
    /// ```
    /// use rusty_machine::stats::dist::Multinomial;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a multinomial distribution for two four sided dice roles.
    /// let multi = Multinomial::new(20, vec![0.25, 0.75]);
    ///
    /// let pdf_nothing = multi.pdf(vec![0, 1]);
    /// assert_eq!(pdf_nothing, 0.0);
    /// ```
    fn pdf(&self, x: Vec<u32>) -> f64 {
        assert!(x.len() == self.p_vals.len(), "Input to pdf must have same length as p_vals.");
        if x.iter().sum::<u32>() != self.experiments {
            return 0.0
        }

        let mut max_index = 0;
        let mut max:u64 = 0;
        for (index, xi) in x.iter().enumerate() {
            if *xi as u64 > max {
                max = *xi as u64;
                max_index = index;
            }
        }

        let mut left = (max+1..self.experiments as u64 + 1).product::<u64>();
        for (index, xi) in x.iter().enumerate() {
            if index != max_index {
                for i in 2..*xi as u64 + 1 {
                    left /= i;
                }
            }
        }
        let right:f64 = self.p_vals.iter().enumerate().map(|(i, p)|p.powi(x[i] as i32)).product::<f64>();
        return left as f64 * right;

    }

    /// The log pdf of the multinomial distribution.
    ///
    /// Implemented simply as the result of calling the log on the pdf function
    fn logpdf(&self, x: Vec<u32>) -> f64 {
        // There's not really a simpler form than just taking the ln of the result.
        // For large N, the whole thing can be approximated by the normal distribution, but
        // that won't really help us here.  Further reading:
        // http://mathworld.wolfram.com/MultinomialDistribution.html
        return self.pdf(x).ln()
    }

    /// The cdf of the multinomial distribution.
    ///
    /// This function is
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::stats::dist::Multinomial;
    /// use rusty_machine::stats::dist::Distribution;
    ///
    /// // Construct a multinomial with 200 trials with probability of success 1/4.
    /// let multi = Multinomial::new(20, vec![0.25, 0.75]);
    ///
    /// let cdf_five = multi.cdf(vec![20, 5]);
    ///
    /// assert_eq!(cdf_five, 3.813027433352545e-6);
    /// ```
    fn cdf(&self, x: Vec<u32>) -> f64 {
        let mut current = vec![0; x.len()];
        let mut total = 0.0;
        fn distribute_remaining(index: usize, current: &mut Vec<u32>, remaining: u32, limit: u32, total: &mut f64, me: &Multinomial, x: &Vec<u32>) {
            let limit_after = limit as i32 - x[index] as i32;
            let lower = max(0, remaining as i32 - limit_after) as u32;
            let upper = min(x[index], remaining);
            if index < x.len() - 1 {
                for assigned_here in lower..upper + 1 {
                    current[index] = assigned_here;
                    distribute_remaining(index + 1, current, remaining - assigned_here, limit - x[index], total, me, x);
                }
            } else {
                current[index] = remaining;
                *total += me.pdf(current.clone());
            }
        }
        distribute_remaining(0, &mut current, self.experiments, x.iter().sum(), &mut total, self, &x);
        return total
    }
}

impl Sample<Vec<u64>> for Multinomial {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Vec<u64> {
        self.ind_sample(rng)
    }
}

impl IndependentSample<Vec<u64>> for Multinomial {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> Vec<u64> {
        let p_len = self.p_vals.len();
        let mut result = Vec::with_capacity(p_len);
        let mut n = self.experiments as i64;
        for index in 0..p_len - 1 {
            let b = Binomial::new(n as u64, self.p_vals[index]);
            let c = b.ind_sample(rng);
            result.push(c);
            n -= c as i64;
            if n < 0 {
                break;
            }
        }
        if n > 0 {
            result.push(n as u64);
        } else {
            result.push(0);
        }
        return result;
    }
}
