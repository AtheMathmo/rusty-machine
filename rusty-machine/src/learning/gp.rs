//! Gaussian Processes
//!
//! Provides implementation of gaussian process regression.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::gp;
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::linalg::vector::Vector;
//!
//! let mut gaussp = gp::GaussianProcess::default();
//! gaussp.noise = 10f64;
//!
//! let train_data = Matrix::new(10,1,vec![0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]);
//! let target = Vector::new(vec![0.,1.,2.,3.,4.,4.,3.,2.,1.,0.]);
//!
//! gaussp.train(&train_data, &target);
//!
//! let test_data = Matrix::new(5,1,vec![2.3,4.4,5.1,6.2,7.1]);
//!
//! let outputs = gaussp.predict(&test_data);
//! ```
//! Alternatively one could use `gaussp.get_posterior()` which would return both
//! the predictive mean and covariance. However, this is likely to change in
//! a future release.

use learning::toolkit::kernel::{Kernel, SquaredExp};
use learning::SupModel;
use linalg::matrix::Matrix;
use linalg::vector::Vector;

/// Trait for GP mean functions.
pub trait MeanFunc {
    /// Compute the mean function applied elementwise to a matrix.
    fn func(&self, x: Matrix<f64>) -> Vector<f64>;
}

/// Constant mean function
#[derive(Clone, Copy, Debug)]
pub struct ConstMean {
    a: f64,
}

/// Constructs the zero function.
impl Default for ConstMean {
    fn default() -> ConstMean {
        ConstMean { a: 0f64 }
    }
}

impl MeanFunc for ConstMean {
    fn func(&self, x: Matrix<f64>) -> Vector<f64> {
        Vector::zeros(x.rows()) + self.a
    }
}

/// Gaussian Process struct
///
/// Gaussian process with generic kernel and deterministic mean function.
/// Can be used for gaussian process regression with noise.
/// Currently does not support classification.
#[derive(Debug)]
pub struct GaussianProcess<T: Kernel, U: MeanFunc> {
    ker: T,
    mean: U,
    /// The observation noise of the GP.
    pub noise: f64,
    alpha: Option<Vector<f64>>,
    train_mat: Option<Matrix<f64>>,
    train_data: Option<Matrix<f64>>,
}

/// Construct a default Gaussian Process
///
/// The defaults are:
///
/// - Squared Exponential kernel.
/// - Zero-mean function.
/// - Zero noise.
///
/// Note that zero noise can often lead to numerical instability.
/// A small value for the noise may be a better alternative.
impl Default for GaussianProcess<SquaredExp, ConstMean> {
    fn default() -> GaussianProcess<SquaredExp, ConstMean> {
        GaussianProcess {
            ker: SquaredExp::default(),
            mean: ConstMean::default(),
            noise: 0f64,
            train_mat: None,
            train_data: None,
            alpha: None,
        }
    }
}

impl<T: Kernel, U: MeanFunc> GaussianProcess<T, U> {
    /// Construct a new Gaussian Process.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gp;
    /// use rusty_machine::learning::toolkit::kernel;
    ///
    /// let ker = kernel::SquaredExp::default();
    /// let mean = gp::ConstMean::default();
    /// let gaussp = gp::GaussianProcess::new(ker, mean, 1e-3f64);
    /// ```
    pub fn new(ker: T, mean: U, noise: f64) -> GaussianProcess<T, U> {
        GaussianProcess {
            ker: ker,
            mean: mean,
            noise: noise,
            train_mat: None,
            train_data: None,
            alpha: None,
        }
    }

    /// Construct a kernel matrix
    fn ker_mat(&self, m1: &Matrix<f64>, m2: &Matrix<f64>) -> Matrix<f64> {
        assert_eq!(m1.cols(), m2.cols());
        let cols = m1.cols();

        let dim1 = m1.rows();
        let dim2 = m2.rows();

        let mut ker_data = Vec::with_capacity(dim1 * dim2);

        for i in 0..dim1 {
            for j in 0..dim2 {
                ker_data.push(self.ker.kernel(&m1.data()[i * cols..(i + 1) * cols],
                                              &m2.data()[j * cols..(j + 1) * cols]));
            }
        }

        Matrix::new(dim1, dim2, ker_data)
    }
}

impl<T: Kernel, U: MeanFunc> SupModel<Matrix<f64>, Vector<f64>> for GaussianProcess<T, U> {
    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {

        // Messy referencing for succint syntax
        if let (&Some(ref alpha), &Some(ref t_data)) = (&self.alpha, &self.train_data) {
            let mean = self.mean.func(inputs.clone());

            let post_mean = self.ker_mat(inputs, t_data) * alpha;

            return mean + post_mean;

        }

        panic!("The model has not been trained.");
    }

    /// Train the model using data and outputs.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let noise_mat = Matrix::identity(inputs.rows()) * self.noise;

        let ker_mat = self.ker_mat(inputs, inputs);

        let train_mat =
            (ker_mat + noise_mat).cholesky().expect("Could not compute Cholesky decomposition.");

        let x = solve_l_triangular(&train_mat, &(targets - self.mean.func(inputs.clone())));
        let alpha = solve_u_triangular(&train_mat.transpose(), &x);

        self.train_mat = Some(train_mat);
        self.train_data = Some(inputs.clone());
        self.alpha = Some(alpha);
    }
}

impl<T: Kernel, U: MeanFunc> GaussianProcess<T, U> {
    /// Compute the posterior distribution [UNSTABLE]
    ///
    /// Requires the model to be trained first.
    ///
    /// Outputs the posterior mean and covariance matrix.
    pub fn get_posterior(&self, inputs: &Matrix<f64>) -> (Vector<f64>, Matrix<f64>) {
        if let (&Some(ref t_mat), &Some(ref alpha), &Some(ref t_data)) = (&self.train_mat,
                                                                          &self.alpha,
                                                                          &self.train_data) {
            let mean = self.mean.func(inputs.clone());

            let post_mean = mean + self.ker_mat(inputs, t_data) * alpha;

            let test_mat = self.ker_mat(inputs, t_data);
            let mut var_data = Vec::with_capacity(inputs.rows() * inputs.cols());
            for i in 0..test_mat.rows() {
                let test_point = Vector::new(test_mat.select_rows(&[i]).into_vec());
                var_data.append(&mut solve_l_triangular(t_mat, &test_point).into_vec());
            }

            let v_mat = Matrix::new(test_mat.rows(), test_mat.cols(), var_data);

            let post_var = self.ker_mat(inputs, inputs) - &v_mat * v_mat.transpose();

            return (post_mean, post_var);
        }

        panic!("The model has not been trained.");
    }
}

/// Solves an upper triangular linear system.
fn solve_u_triangular(mat: &Matrix<f64>, y: &Vector<f64>) -> Vector<f64> {
    assert!(mat.cols() == y.size(),
            "Matrix and Vector dimensions do not agree.");

    let mut x = vec![0.; y.size()];


    x[y.size() - 1] = y[y.size() - 1] / mat[[y.size() - 1, y.size() - 1]];

    for i in (0..y.size() - 1).rev() {
        let mut holding_u_sum = 0.;
        for j in (i + 1..y.size()).rev() {
            holding_u_sum += mat.data()[i * mat.cols() + j] * x[j];
        }

        x[i] = (y[i] - holding_u_sum) / mat.data()[i * (mat.cols() + 1)];
    }

    Vector::new(x)
}

/// Solves a lower triangular linear system.
fn solve_l_triangular(mat: &Matrix<f64>, y: &Vector<f64>) -> Vector<f64> {
    assert!(mat.cols() == y.size(),
            "Matrix and Vector dimensions do not agree.");

    let mut x = vec![0.; y.size()];


    x[0] = y[0] / mat[[0, 0]];

    for i in 1..y.size() {
        let mut holding_l_sum = 0.;
        for (j, x_item) in x.iter().enumerate().take(i) {
            holding_l_sum += mat.data()[i * mat.cols() + j] * x_item;
        }

        x[i] = (y[i] - holding_l_sum) / mat.data()[i * (mat.cols() + 1)];
    }

    Vector::new(x)
}
