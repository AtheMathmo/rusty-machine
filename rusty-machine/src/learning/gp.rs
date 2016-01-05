use learning::toolkit::kernel::{Kernel, SquaredExp};
use learning::SupModel;
use linalg::matrix::Matrix;

pub trait MeanFunc {
    fn func(&self, x: Matrix<f64>) -> Matrix<f64>;
}

/// Constant mean function
pub struct ConstMean {
    a: f64,
}

impl Default for ConstMean {
    fn default() -> ConstMean {
        ConstMean { a: 0f64 }
    }
}

impl MeanFunc for ConstMean {
    fn func(&self, x: Matrix<f64>) -> Matrix<f64> {
        Matrix::ones(x.rows(), x.cols()) * self.a
    }
}

/// Gaussian Process struct
///
/// Gaussian process with generic kernel and deterministic mean function.
/// Can be used for gaussian process regression with noise.
/// Currently does not support classification.
///
/// # Note
///
/// The 
pub struct GaussianProcess<T: Kernel, U: MeanFunc> {
    ker: T,
    mean: U,
    noise: f64,
    train_data: Option<Matrix<f64>>,
    train_output: Option<Matrix<f64>>,
    train_mat: Option<Matrix<f64>>,
}

impl Default for GaussianProcess<SquaredExp, ConstMean> {
    /// Construct a default Gaussian Process
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::gp;
    /// 
    /// let gp = gp::GaussianProcess::default();
    /// ```
    fn default() -> GaussianProcess<SquaredExp, ConstMean> {
        GaussianProcess {
            ker: SquaredExp::default(),
            mean: ConstMean::default(),
            noise: 1f64,
            train_data: None,
            train_output: None,
            train_mat: None,
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
            train_data: None,
            train_output: None,
            train_mat: None,
        }
    }

    /// Construct a kernel matrix
    fn ker_mat(&self, m1: &Matrix<f64>, m2: &Matrix<f64>) -> Matrix<f64> {
        assert_eq!(m1.cols(), m2.cols());
        let dim1 = m1.rows();
        let dim2 = m2.rows();

        let mut ker_data = Vec::with_capacity(dim1 * dim2);

        for i in 0..dim1 {
            for j in 0..dim2 {
                ker_data.push(self.ker.kernel(&m1.data[i * m1.cols()..(i + 1) * m1.cols()],
                                              &m2.data[j * m2.cols()..(j + 1) * m2.cols()]));
            }
        }

        Matrix::new(dim1, dim2, ker_data)
    }
}

impl<T: Kernel, U: MeanFunc> SupModel<Matrix<f64>, Matrix<f64>> for GaussianProcess<T, U> {
    /// Predict output from data.
    fn predict(&self, data: &Matrix<f64>) -> Matrix<f64> {
        let mean = self.mean.func(data.clone());

        // Messy referencing for succint syntax
        if let (&Some(ref t_data), &Some(ref t_mat), &Some(ref t_out)) = (&self.train_data,
                                                                          &self.train_mat,
                                                                          &self.train_output) {
            let test_mat = self.ker_mat(data, t_data) * t_mat *
                           (t_out - self.mean.func(t_data.clone()));
            return mean + test_mat;
        }

        panic!("The model has not been trained.");
    }

    /// Train the model using data and outputs.
    fn train(&mut self, data: &Matrix<f64>, value: &Matrix<f64>) {
        let noise_mat = Matrix::identity(data.rows()) * self.noise;

        let ker_mat = self.ker_mat(data, data);

        let train_mat = (ker_mat + noise_mat).inverse();

        self.train_mat = Some(train_mat);
        self.train_data = Some(data.clone());
        self.train_output = Some(value.clone());
    }
}

impl<T: Kernel, U: MeanFunc> GaussianProcess<T,U> {

	/// Compute the posterior distribution
	///
	/// Requires the model to be trained first. 
	/// _Note that this is a messy compromise as GPs do not
	/// fit the SupModel trait as is currently implemented._
	pub fn get_posterior(&self, data: &Matrix<f64>) -> (Matrix<f64>, Matrix<f64>) {
		let mean = self.mean.func(data.clone());

        // Messy referencing for succint syntax
        if let (&Some(ref t_data), &Some(ref t_mat), &Some(ref t_out)) = (&self.train_data,
                                                                          &self.train_mat,
                                                                          &self.train_output) {
            let test_mat = self.ker_mat(data, t_data) * t_mat;            
            let post_mean = mean + &test_mat * (t_out - self.mean.func(t_data.clone()));

            let post_cov = self.ker_mat(data, data) - test_mat * self.ker_mat(t_data, data);

            return (post_mean, post_cov);
        }

        panic!("The model has not been trained.");
	}
}