//! Module for kernels
//!
//! Currently used within Gaussian Processes and SVMs.

use std::ops::{Add, Mul};

use linalg::Vector;
use linalg::norm::{Euclidean, VectorNorm, VectorMetric};
use rulinalg::utils;

/// The Kernel trait
///
/// Requires a function mapping two vectors to a scalar.
pub trait Kernel {
    /// The kernel function.
    ///
    /// Takes two equal length slices and returns a scalar.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64;
}

/// The sum of two kernels
///
/// This struct should not be directly instantiated but instead
/// is created when we add two kernels together.
///
/// Note that it will be more efficient to implement the final kernel
/// manually yourself. However this provides an easy mechanism to test
/// different combinations.
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::toolkit::kernel::{Kernel, Polynomial, HyperTan, KernelArith};
///
/// let poly_ker = Polynomial::new(1f64,2f64,3f64);
/// let hypert_ker = HyperTan::new(1f64,2.5);
///
/// let poly_plus_hypert_ker = KernelArith(poly_ker) + KernelArith(hypert_ker);
///
/// println!("{0}", poly_plus_hypert_ker.kernel(&[1f64,2f64,3f64],
///                                             &[3f64,1f64,2f64]));
/// ```
#[derive(Debug)]
pub struct KernelSum<T, U>
    where T: Kernel,
          U: Kernel
{
    k1: T,
    k2: U,
}

/// Computes the sum of the two associated kernels.
impl<T, U> Kernel for KernelSum<T, U>
    where T: Kernel,
          U: Kernel
{
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        self.k1.kernel(x1, x2) + self.k2.kernel(x1, x2)
    }
}

/// The pointwise product of two kernels
///
/// This struct should not be directly instantiated but instead
/// is created when we multiply two kernels together.
///
/// Note that it will be more efficient to implement the final kernel
/// manually yourself. However this provides an easy mechanism to test
/// different combinations.
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::toolkit::kernel::{Kernel, Polynomial, HyperTan, KernelArith};
///
/// let poly_ker = Polynomial::new(1f64,2f64,3f64);
/// let hypert_ker = HyperTan::new(1f64,2.5);
///
/// let poly_plus_hypert_ker = KernelArith(poly_ker) * KernelArith(hypert_ker);
///
/// println!("{0}", poly_plus_hypert_ker.kernel(&[1f64,2f64,3f64],
///                                             &[3f64,1f64,2f64]));
/// ```
#[derive(Debug)]
pub struct KernelProd<T, U>
    where T: Kernel,
          U: Kernel
{
    k1: T,
    k2: U,
}

/// Computes the product of the two associated kernels.
impl<T, U> Kernel for KernelProd<T, U>
    where T: Kernel,
          U: Kernel
{
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        self.k1.kernel(x1, x2) * self.k2.kernel(x1, x2)
    }
}

/// A wrapper tuple struct used for kernel arithmetic
#[derive(Debug)]
pub struct KernelArith<K: Kernel>(pub K);

impl<T: Kernel, U: Kernel> Add<KernelArith<T>> for KernelArith<U> {
    type Output = KernelSum<U, T>;

    fn add(self, ker: KernelArith<T>) -> KernelSum<U, T> {
        KernelSum {
            k1: self.0,
            k2: ker.0,
        }
    }
}

impl<T: Kernel, U: Kernel> Mul<KernelArith<T>> for KernelArith<U> {
    type Output = KernelProd<U, T>;

    fn mul(self, ker: KernelArith<T>) -> KernelProd<U, T> {
        KernelProd {
            k1: self.0,
            k2: ker.0,
        }
    }
}

/// The Linear Kernel
///
/// k(x,y) = x<sup>T</sup>y + c
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    /// Constant term added to inner product.
    pub c: f64,
}

impl Linear {
    /// Constructs a new Linear Kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// let ker = kernel::Linear::new(5.0);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(c: f64) -> Linear {
        Linear { c: c }
    }
}

/// Constructs the default Linear Kernel
///
/// The defaults are:
///
/// - c = 0
impl Default for Linear {
    fn default() -> Linear {
        Linear { c: 0f64 }
    }
}

impl Kernel for Linear {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        utils::dot(x1, x2) + self.c
    }
}

/// The Polynomial Kernel
///
/// k(x,y) = (αx<sup>T</sup>y + c)<sup>d</sup>
#[derive(Clone, Copy, Debug)]
pub struct Polynomial {
    /// Scaling of the inner product.
    pub alpha: f64,
    /// Constant added to inner product.
    pub c: f64,
    /// The power to raise the sum to.
    pub d: f64,
}

impl Polynomial {
    /// Constructs a new Polynomial Kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Constructs a new polynomial with alpha = 1, c = 0, d = 2.
    /// let ker = kernel::Polynomial::new(1.0, 0.0, 2.0);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(alpha: f64, c: f64, d: f64) -> Polynomial {
        Polynomial {
            alpha: alpha,
            c: c,
            d: d,
        }
    }
}

/// Construct a new polynomial kernel.
///
/// The defaults are:
///
/// - alpha = 1
/// - c = 0
/// - d = 1
impl Default for Polynomial {
    fn default() -> Polynomial {
        Polynomial {
            alpha: 1f64,
            c: 0f64,
            d: 1f64,
        }
    }
}

impl Kernel for Polynomial {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        (self.alpha * utils::dot(x1, x2) + self.c).powf(self.d)
    }
}

/// Squared exponential kernel
///
/// Equivalently a gaussian function.
///
/// k(x,y) = A _exp_(-||x-y||<sup>2</sup> / 2l<sup>2</sup>)
///
/// Where A is the amplitude and l the length scale.
#[derive(Clone, Copy, Debug)]
pub struct SquaredExp {
    /// The length scale of the kernel.
    pub ls: f64,
    /// The amplitude of the kernel.
    pub ampl: f64,
}

impl SquaredExp {
    /// Construct a new squared exponential kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Construct a kernel with lengthscale 2 and amplitude 1.
    /// let ker = kernel::SquaredExp::new(2f64, 1f64);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(ls: f64, ampl: f64) -> SquaredExp {
        SquaredExp {
            ls: ls,
            ampl: ampl,
        }
    }
}

/// Constructs the default Squared Exp kernel.
///
/// The defaults are:
///
/// - ls = 1
/// - ampl = 1
impl Default for SquaredExp {
    fn default() -> SquaredExp {
        SquaredExp {
            ls: 1f64,
            ampl: 1f64,
        }
    }
}

impl Kernel for SquaredExp {
    /// The squared exponential kernel function.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());

        let diff = Vector::new(x1.to_vec()) - Vector::new(x2.to_vec());

        let x = -diff.dot(&diff) / (2f64 * self.ls * self.ls);
        (self.ampl * x.exp())
    }
}

/// The Exponential Kernel
///
/// k(x,y) = A _exp_(-||x-y|| / 2l<sup>2</sup>)
///
/// Where A is the amplitude and l is the length scale.
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    /// The length scale of the kernel.
    pub ls: f64,
    /// The amplitude of the kernel.
    pub ampl: f64,
}

impl Exponential {
    /// Construct a new squared exponential kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Construct a kernel with lengthscale 2 and amplitude 1.
    /// let ker = kernel::Exponential::new(2f64, 1f64);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(ls: f64, ampl: f64) -> Exponential {
        Exponential {
            ls: ls,
            ampl: ampl,
        }
    }
}

/// Constructs the default Exponential kernel.
///
/// The defaults are:
///
/// - ls = 1
/// - amplitude = 1
impl Default for Exponential {
    fn default() -> Exponential {
        Exponential {
            ls: 1f64,
            ampl: 1f64,
        }
    }
}

impl Kernel for Exponential {
    /// The squared exponential kernel function.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());

        let diff = Vector::new(x1.to_vec()) - Vector::new(x2.to_vec());

        let x = -Euclidean.norm(&diff) / (2f64 * self.ls * self.ls);
        (self.ampl * x.exp())
    }
}

/// The Hyperbolic Tangent Kernel.
///
/// ker(x,y) = _tanh_(αx<sup>T</sup>y + c)
#[derive(Clone, Copy, Debug)]
pub struct HyperTan {
    /// The scaling of the inner product.
    pub alpha: f64,
    /// The constant to add to the inner product.
    pub c: f64,
}

impl HyperTan {
    /// Constructs a new Hyperbolic Tangent Kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Construct a kernel with alpha = 1, c = 2.
    /// let ker = kernel::HyperTan::new(1.0, 2.0);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(alpha: f64, c: f64) -> HyperTan {
        HyperTan {
            alpha: alpha,
            c: c,
        }
    }
}

/// Constructs a default Hyperbolic Tangent Kernel.
///
/// The defaults are:
///
/// - alpha = 1
/// - c = 0
impl Default for HyperTan {
    fn default() -> HyperTan {
        HyperTan {
            alpha: 1f64,
            c: 0f64,
        }
    }
}

impl Kernel for HyperTan {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        (self.alpha * utils::dot(x1, x2) + self.c).tanh()
    }
}

/// The Multiquadric Kernel.
///
/// k(x,y) = _sqrt_(||x-y||<sup>2</sup> + c<sup>2</sup>)
#[derive(Clone, Copy, Debug)]
pub struct Multiquadric {
    /// Constant added to square of difference.
    pub c: f64,
}

impl Multiquadric {
    /// Constructs a new Multiquadric Kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Construct a kernel with c = 2.
    /// let ker = kernel::Multiquadric::new(2.0);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(c: f64) -> Multiquadric {
        Multiquadric { c: c }
    }
}

/// Constructs a default Multiquadric Kernel.
///
/// The defaults are:
///
/// - c = 0
impl Default for Multiquadric {
    fn default() -> Multiquadric {
        Multiquadric { c: 0f64 }
    }
}

impl Kernel for Multiquadric {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());

        Euclidean.metric(&(x1.into()), &(x2.into())).hypot(self.c)
    }
}

/// The Rational Quadratic Kernel.
///
/// k(x,y) = (1 + ||x-y||<sup>2</sup> / (2αl<sup>2</sup>))<sup>-α</sup>
#[derive(Clone, Copy, Debug)]
pub struct RationalQuadratic {
    /// Controls inverse power and difference scale.
    pub alpha: f64,
    /// Length scale controls scale of difference.
    pub ls: f64,
}

impl RationalQuadratic {
    /// Constructs a new Rational Quadratic Kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::toolkit::kernel;
    /// use rusty_machine::learning::toolkit::kernel::Kernel;
    ///
    /// // Construct a kernel with alpha = 2, ls = 2.
    /// let ker = kernel::RationalQuadratic::new(2.0, 2.0);
    ///
    /// println!("{0}", ker.kernel(&[1.,2.,3.], &[3.,4.,5.]));
    /// ```
    pub fn new(alpha: f64, ls: f64) -> RationalQuadratic {
        RationalQuadratic {
            alpha: alpha,
            ls: ls,
        }
    }
}

/// The default Rational Qaudratic Kernel.
///
/// The defaults are:
///
/// - alpha = 1
/// - ls = 1
impl Default for RationalQuadratic {
    fn default() -> RationalQuadratic {
        RationalQuadratic {
            alpha: 1f64,
            ls: 1f64,
        }
    }
}

impl Kernel for RationalQuadratic {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let diff = Vector::new(x1.to_vec()) - Vector::new(x2.to_vec());

        (1f64 + diff.dot(&diff) / (2f64 * self.alpha * self.ls * self.ls)).powf(-self.alpha)
    }
}
