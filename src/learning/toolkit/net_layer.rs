//!Neural Network Layers

use linalg::{Matrix, MatrixSlice};
use linalg::BaseSlice;

use learning::toolkit::activ_fn::ActivationFunc;

use rand::thread_rng;
use rand::distributions::Sample;
use rand::distributions::normal::Normal;

/// Trait for neural net layers
pub trait NetLayer {
	/// The result of propogating data forward through this layer
	fn forward(&self, input: &Matrix<f64>, params: MatrixSlice<f64>) -> Matrix<f64>;

	/// The gradient of the output of this layer with respect to its input
	fn backward(&self, out_grad: &Matrix<f64>, params: MatrixSlice<f64>) -> Matrix<f64>;

	/// The default value of the parameters of this layer before training
	fn default_params(&self) -> Vec<f64>;

	/// The number of parameters used by this layer
	fn num_params(&self) -> usize;

	/// The shape of the parameters used by this layer
	fn param_shape(&self) -> (usize, usize);
}

/// Linear network layer
///
/// The parameters are a matrix of weights of size O x I
/// where O is the dimensionality of the output and I the dimensionality of the input
#[derive(Debug, Clone, Copy)]
pub struct Linear {
	/// The number of dimensions of the input
	input_size: usize,
	/// The number of dimensions of the output
	output_size: usize,
}

impl Linear {
	/// Construct a Linear layer
	pub fn new(input_size: usize, output_size: usize) -> Linear {
		Linear{input_size: input_size, output_size: output_size}
	}
}

impl NetLayer for Linear {
	/// Computes a matrix product
	///
	/// input should have dimensions N x I
	/// where N is the number of samples and I is the dimensionality of the input
	fn forward(&self, input: &Matrix<f64>, params: MatrixSlice<f64>) -> Matrix<f64> {
		assert_eq!(input.cols(), params.rows());
		input * &params
	}

	fn backward(&self, out_grad: &Matrix<f64>, params: MatrixSlice<f64>) -> Matrix<f64> {
		assert_eq!(out_grad.cols(), params.cols());
		out_grad * params.into_matrix().transpose()
	}

	/// Initializes weights using Xavier initialization
	///
	/// weights draw from gaussian distribution with 0 mean and variance 2/(input_size+output_size)
	fn default_params(&self) -> Vec<f64> {
		let mut distro = Normal::new(0.0, (2.0/(self.input_size+self.output_size) as f64).sqrt());
		let mut rng = thread_rng();

		(0..self.input_size*self.output_size).map(|_| distro.sample(&mut rng))
											 .collect()
	}

	fn num_params(&self) -> usize {
		self.output_size * self.input_size
	}

	fn param_shape(&self) -> (usize, usize) {
		(self.input_size, self.output_size)
	}
}

impl<T: ActivationFunc> NetLayer for T {
	/// Applys the activation function to each element of the input
	fn forward(&self, input: &Matrix<f64>, _: MatrixSlice<f64>) -> Matrix<f64> {
		input.clone().apply(&T::func)
	}

	fn backward(&self, out_grad: &Matrix<f64>, _: MatrixSlice<f64>) -> Matrix<f64> {
		out_grad.clone().apply(&T::func_grad)
	}

	fn default_params(&self) -> Vec<f64> {
		vec![]
	}

	fn num_params(&self) -> usize {
		0
	}

	fn param_shape(&self) -> (usize, usize) {
		(0, 0)
	}
}