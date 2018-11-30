//! Neural Network Layers

use linalg::{BaseMatrix, Matrix, MatrixSlice};

use learning::error::{Error, ErrorKind};
use learning::toolkit::activ_fn::ActivationFunc;
use learning::LearningResult;

use rand::distributions::normal::Normal;
use rand::distributions::Sample;
use rand::thread_rng;

use std::fmt::Debug;

/// Trait for neural net layers
pub trait NetLayer: Debug {
    /// The result of propogating data forward through this layer
    fn forward(&self, input: &Matrix<f64>, params: MatrixSlice<f64>)
        -> LearningResult<Matrix<f64>>;

    /// The gradient of the output of this layer with respect to its input
    fn back_input(
        &self,
        out_grad: &Matrix<f64>,
        input: &Matrix<f64>,
        output: &Matrix<f64>,
        params: MatrixSlice<f64>,
    ) -> Matrix<f64>;

    /// The gradient of the output of this layer with respect to its parameters
    fn back_params(
        &self,
        out_grad: &Matrix<f64>,
        input: &Matrix<f64>,
        output: &Matrix<f64>,
        params: MatrixSlice<f64>,
    ) -> Matrix<f64>;

    /// The default value of the parameters of this layer before training
    fn default_params(&self) -> Vec<f64>;

    /// The shape of the parameters used by this layer
    fn param_shape(&self) -> (usize, usize);

    /// The number of parameters used by this layer
    fn num_params(&self) -> usize {
        let shape = self.param_shape();
        shape.0 * shape.1
    }
}

/// Linear network layer
///
/// Represents a fully connected layer with optional bias term
///
/// The parameters are a matrix of weights of size I x N
/// where N is the dimensionality of the output and I the dimensionality of the input
#[derive(Debug, Clone, Copy)]
pub struct Linear {
    /// The number of dimensions of the input
    input_size: usize,
    /// The number of dimensions of the output
    output_size: usize,
    /// Whether or not to include a bias term
    has_bias: bool,
}

impl Linear {
    /// Construct a new Linear layer
    pub fn new(input_size: usize, output_size: usize) -> Linear {
        Linear {
            input_size: input_size + 1,
            output_size: output_size,
            has_bias: true,
        }
    }

    /// Construct a Linear layer without a bias term
    pub fn without_bias(input_size: usize, output_size: usize) -> Linear {
        Linear {
            input_size: input_size,
            output_size: output_size,
            has_bias: false,
        }
    }
}

fn remove_first_col(mat: Matrix<f64>) -> Matrix<f64> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut data = mat.into_vec();

    let len = data.len();
    let mut del = 0;
    {
        let v = &mut *data;

        for i in 0..len {
            if i % cols == 0 {
                del += 1;
            } else if del > 0 {
                v[i - del] = v[i];
            }
        }
    }
    if del > 0 {
        data.truncate(len - del);
    }
    Matrix::new(rows, cols - 1, data)
}

impl NetLayer for Linear {
    /// Computes a matrix product
    ///
    /// input should have dimensions N x I
    /// where N is the number of samples and I is the dimensionality of the input
    fn forward(
        &self,
        input: &Matrix<f64>,
        params: MatrixSlice<f64>,
    ) -> LearningResult<Matrix<f64>> {
        if self.has_bias {
            if input.cols() + 1 != params.rows() {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "The input had the wrong number of columns",
                ))
            } else {
                Ok(&Matrix::ones(input.rows(), 1).hcat(input) * &params)
            }
        } else {
            if input.cols() != params.rows() {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "The input had the wrong number of columns",
                ))
            } else {
                Ok(input * &params)
            }
        }
    }

    fn back_input(
        &self,
        out_grad: &Matrix<f64>,
        _: &Matrix<f64>,
        _: &Matrix<f64>,
        params: MatrixSlice<f64>,
    ) -> Matrix<f64> {
        debug_assert_eq!(out_grad.cols(), params.cols());
        let gradient = out_grad * &params.transpose();
        if self.has_bias {
            remove_first_col(gradient)
        } else {
            gradient
        }
    }

    fn back_params(
        &self,
        out_grad: &Matrix<f64>,
        input: &Matrix<f64>,
        _: &Matrix<f64>,
        _: MatrixSlice<f64>,
    ) -> Matrix<f64> {
        debug_assert_eq!(input.rows(), out_grad.rows());
        if self.has_bias {
            &Matrix::ones(input.rows(), 1).hcat(input).transpose() * out_grad
        } else {
            &input.transpose() * out_grad
        }
    }

    /// Initializes weights using Xavier initialization
    ///
    /// weights drawn from gaussian distribution with 0 mean and variance 2/(input_size+output_size)
    fn default_params(&self) -> Vec<f64> {
        let mut distro = Normal::new(
            0.0,
            (2.0 / (self.input_size + self.output_size) as f64).sqrt(),
        );
        let mut rng = thread_rng();

        (0..self.input_size * self.output_size)
            .map(|_| distro.sample(&mut rng))
            .collect()
    }

    fn param_shape(&self) -> (usize, usize) {
        (self.input_size, self.output_size)
    }
}

impl<T: ActivationFunc> NetLayer for T {
    /// Applies the activation function to each element of the input
    fn forward(&self, input: &Matrix<f64>, _: MatrixSlice<f64>) -> LearningResult<Matrix<f64>> {
        let mut output = Vec::with_capacity(input.rows() * input.cols());
        for val in input.data() {
            output.push(T::func(*val));
        }
        Ok(Matrix::new(input.rows(), input.cols(), output))
    }

    fn back_input(
        &self,
        out_grad: &Matrix<f64>,
        _: &Matrix<f64>,
        output: &Matrix<f64>,
        _: MatrixSlice<f64>,
    ) -> Matrix<f64> {
        let mut in_grad = Vec::with_capacity(output.rows() * output.cols());
        for (y, g) in output.data().iter().zip(out_grad.data()) {
            in_grad.push(T::func_grad_from_output(*y) * g);
        }
        Matrix::new(output.rows(), output.cols(), in_grad)
    }

    fn back_params(
        &self,
        _: &Matrix<f64>,
        _: &Matrix<f64>,
        _: &Matrix<f64>,
        _: MatrixSlice<f64>,
    ) -> Matrix<f64> {
        Matrix::new(0, 0, Vec::new())
    }

    fn default_params(&self) -> Vec<f64> {
        Vec::new()
    }

    fn param_shape(&self) -> (usize, usize) {
        (0, 0)
    }
}
