use learning::toolkit::kernel::{Kernel, SquaredExp};
use learning::SupModel;
use linalg::matrix::Matrix;

trait MeanFunc {
    fn func(&self, x: f64) -> f64;
}

struct ConstMean {
    a: f64,
}

impl Default for ConstMean {
    fn default() -> ConstMean {
        ConstMean { a: 0f64 }
    }
}

impl MeanFunc for ConstMean {
    fn func(&self, _: f64) -> f64 {
        self.a
    }
}


struct GaussianProcess<T: Kernel, U: MeanFunc> {
    ker: T,
    mean: U,
    noise: f64,
    train_data: Option<Matrix<f64>>,
    train_output: Option<Matrix<f64>>,
    train_mat: Option<Matrix<f64>>,
}

impl Default for GaussianProcess<SquaredExp, ConstMean> {
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
        	Matrix::new(1,1,vec![1f64])
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
