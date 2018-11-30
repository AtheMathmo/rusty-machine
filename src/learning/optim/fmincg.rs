//! Module for the fmincg optimization algorithm.
//!
//! This algorithm was taken from Andrew Ng's coursera machine
//! learning course. The function was translated from MATLAB into rust.
//! Original source code can be found [here](http://www.mathworks.com/matlabcentral/fileexchange/42770-logistic-regression-with-regularization-used-to-classify-hand-written-digits/content/Logistic%20Regression%20with%20regularisation/fmincg.m).
//!
//! The attached license permits use and modification for research
//! and education only.
//!
//! Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
//!
//!
//! (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
//!
//! Permission is granted for anyone to copy, use, or modify these
//! programs and accompanying documents for purposes of research or
//! education, provided this copyright notice is retained, and note is
//! made of any changes that have been made.
//!
//! These programs and documents are distributed without any warranty,
//! express or implied.  As the programs were written for research
//! purposes only, they have not been tested to the degree that would be
//! advisable in any important application.  All use of these programs is
//! entirely at the user's own risk.
//!
//! [rusty-machine] Changes made:
//!
//! - Conversion to Rust.
//! - Length hard defaults to the max iterations.

use learning::optim::{OptimAlgorithm, Optimizable};
use linalg::Vector;

use std::cmp;
use std::f64;

/// Conjugate Gradient Descent algorithm
#[derive(Clone, Copy, Debug)]
pub struct ConjugateGD {
    /// Constant in the Wolfe-Powell conditions.
    pub rho: f64,
    /// Constant in the Wolfe-Powell conditions.
    pub sig: f64,
    /// Don't reevaluate within `int` of the limit of the current bracket.
    pub int: f64,
    /// Extrapolate max of `ext` times the current bracket.
    pub ext: f64,
    /// Max of `max` function evaluations per line search
    pub max: usize,
    /// The maximum allowed slope ratio
    pub ratio: f64,

    /// The default number of max iterations.
    pub iters: usize,
}

/// The default Conjugate GD algorithm.
///
/// The defaults are:
///
/// - rho = 0.01
/// - sig = 0.5
/// - int = 0.1
/// - ext = 3
/// - max = 20
/// - ration = 100
/// - iters = 100
impl Default for ConjugateGD {
    fn default() -> ConjugateGD {
        ConjugateGD {
            rho: 0.01,
            sig: 0.5,
            int: 0.1,
            ext: 3.0,
            max: 20,
            ratio: 100.0,
            iters: 100,
        }
    }
}

impl<M: Optimizable> OptimAlgorithm<M> for ConjugateGD {
    fn optimize(
        &self,
        model: &M,
        start: &[f64],
        inputs: &M::Inputs,
        targets: &M::Targets,
    ) -> Vec<f64> {
        let mut i = 0usize;
        let mut ls_failed = false;

        let (mut f1, vec_df1) = model.compute_grad(start, inputs, targets);
        let mut df1 = Vector::new(vec_df1);

        // The reduction in the function. Can also be specified as part of length
        let red = 1f64;

        let length = self.iters as i32;

        let mut s = -df1.clone();
        let mut d1 = -s.dot(&s);
        let mut z1 = red / (1f64 - d1);

        let mut x = Vector::new(start.to_vec());

        let (mut f2, mut df2): (f64, Vector<f64>);

        while (i as i32) < length.abs() {
            if length > 0 {
                i += 1;
            }

            let (x0, f0) = (x.clone(), f1);

            x += &s * z1;

            let cost = model.compute_grad(x.data(), inputs, targets);
            f2 = cost.0;
            df2 = Vector::new(cost.1);

            if length < 0 {
                i += 1;
            }

            let mut d2 = df2.dot(&s);

            let (mut f3, mut d3, mut z3) = (f1, d1, -z1);

            let mut m = if length > 0 {
                self.max as i32
            } else {
                cmp::min(self.max as i32, -length - (i as i32))
            };

            let mut success = false;
            let mut limit = -1f64;

            loop {
                let mut z2: f64;

                while ((f2 > (f1 + z1 * self.rho * d1)) || (d2 > -self.sig * d1)) && (m > 0i32) {
                    limit = z1;

                    if f2 > f1 {
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                    } else {
                        let a = 6f64 * (f2 - f3) / z3 + 3f64 * (d2 + d3);
                        let b = 3f64 * (f3 - f2) - z3 * (2f64 * d2 + d3);
                        z2 = ((b * b - a * d2 * z3 * z3).sqrt() - b) / a;
                    }

                    if z2.is_nan() || z2.is_infinite() {
                        z2 = z3 / 2f64;
                    }

                    if z2 <= self.int * z3 {
                        if z2 <= (1f64 - self.int) * z3 {
                            z2 = (1f64 - self.int) * z3;
                        }
                    } else if self.int * z3 <= (1f64 - self.int) * z3 {
                        z2 = (1f64 - self.int) * z3;
                    } else {
                        z2 = self.int * z3;
                    }

                    z1 += z2;
                    x += &s * z2;
                    let cost_grad = model.compute_grad(x.data(), inputs, targets);
                    f2 = cost_grad.0;
                    df2 = Vector::new(cost_grad.1);

                    m -= 1i32;
                    if length < 0 {
                        i += 1;
                    }

                    d2 = df2.dot(&s);
                    z3 -= z2;
                }

                if f2 > f1 + z1 * self.rho * d1 || d2 > -self.sig * d1 {
                    break;
                } else if d2 > self.sig * d1 {
                    success = true;
                    break;
                } else if m == 0i32 {
                    break;
                }

                let a = 6f64 * (f2 - f3) / z3 + 3f64 * (d2 + d3);
                let b = 3f64 * (f3 - f2) - z3 * (2f64 * d2 + d3);
                z2 = -d2 * z3 * z3 / (b + (b * b - a * d2 * z3 * z3).sqrt());

                if z2.is_nan() || z2.is_infinite() || z2 < 0f64 {
                    if limit < -0.5 {
                        z2 = z1 * (self.ext - 1f64);
                    } else {
                        z2 = (limit - z1) / 2f64;
                    }
                } else if (limit > -0.5) && (z2 + z1 > limit) {
                    z2 = (limit - z1) / 2f64;
                } else if (limit < -0.5) && (z2 + z1 > z1 * self.ext) {
                    z2 = z1 * (self.ext - 1f64);
                } else if z2 < -z3 * self.int {
                    z2 = -z3 * self.int;
                } else if (limit > -0.5) && (z2 < (limit - z1) * (1f64 - self.int)) {
                    z2 = (limit - z1) * (1f64 - self.int);
                }

                f3 = f2;
                d3 = d2;
                z3 = -z2;
                z1 += z2;
                x += &s * z2;

                let cost_grad = model.compute_grad(x.data(), inputs, targets);
                f2 = cost_grad.0;
                df2 = Vector::new(cost_grad.1);

                m -= 1;
                if length < 0 {
                    i += 1;
                }

                d2 = df2.dot(&s);
            }

            if success {
                f1 = f2;
                s = s * (&df2 - &df1).dot(&df2) / df1.dot(&df1) - &df2;

                df1 = df2;

                d2 = df1.dot(&s);

                if d2 > 0f64 {
                    s = -&df1;
                    d2 = -s.dot(&s);
                }

                let ratio = d1 / (d2 - f64::MIN_POSITIVE);
                if self.ratio < ratio {
                    z1 *= self.ratio;
                } else {
                    z1 *= ratio;
                }

                d1 = d2;
                ls_failed = false;
            } else {
                x = x0;
                f1 = f0;

                if ls_failed || i as i32 > length.abs() {
                    break;
                }

                df1 = df2;

                s = -&df1;
                d1 = -s.dot(&s);

                z1 = 1f64 / (1f64 - d1);
                ls_failed = true;
            }
        }
        x.into_vec()
    }
}
