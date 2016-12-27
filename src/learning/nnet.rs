//! Neural Network module
//!
//! Contains implementation of simple feed forward neural network.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
//! use rusty_machine::learning::toolkit::regularization::Regularization;
//! use rusty_machine::learning::optim::grad_desc::StochasticGD;
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::learning::SupModel;
//!
//! let inputs = Matrix::new(5,3, vec![1.,1.,1.,2.,2.,2.,3.,3.,3.,
//!                                 4.,4.,4.,5.,5.,5.,]);
//! let targets = Matrix::new(5,3, vec![1.,0.,0.,0.,1.,0.,0.,0.,1.,
//!                                     0.,0.,1.,0.,0.,1.]);
//!
//! // Set the layer sizes - from input to output
//! let layers = &[3,5,11,7,3];
//!
//! // Choose the BCE criterion with L2 regularization (`lambda=0.1`).
//! let criterion = BCECriterion::new(Regularization::L2(0.1));
//!
//! // We will just use the default stochastic gradient descent.
//! let mut model = NeuralNet::new(layers, criterion, StochasticGD::default());
//!
//! // Train the model!
//! model.train(&inputs, &targets).unwrap();
//!
//! let test_inputs = Matrix::new(2,3, vec![1.5,1.5,1.5,5.1,5.1,5.1]);
//!
//! // And predict new output from the test inputs
//! let outputs = model.predict(&test_inputs).unwrap();
//! ```
//!
//! The neural networks are specified via a criterion - similar to
//! [Torch](https://github.com/torch/nn/blob/master/doc/criterion.md).
//! The criterions combine an activation function and a cost function.
//!
//! You can define your own criterion by implementing the `Criterion`
//! trait with a concrete `ActivationFunc` and `CostFunc`.

use linalg::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};

use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};
use learning::toolkit::activ_fn;
use learning::toolkit::activ_fn::ActivationFunc;
use learning::toolkit::cost_fn;
use learning::toolkit::cost_fn::CostFunc;
use learning::toolkit::regularization::Regularization;
use learning::optim::{Optimizable, OptimAlgorithm};
use learning::optim::grad_desc::StochasticGD;

use rand::thread_rng;
use rand::distributions::{Sample, range};

/// Neural Network Model
///
/// The Neural Network struct specifies a Criterion and
/// a gradient descent algorithm.
#[derive(Debug)]
pub struct NeuralNet<'a, T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<'a, T>>
{
    base: BaseNeuralNet<'a, T>,
    alg: A,
}

/// Supervised learning for the Neural Network.
///
/// The model is trained using back propagation.
impl<'a, T, A> SupModel<Matrix<f64>, Matrix<f64>> for NeuralNet<'a, T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<'a, T>>
{
    /// Predict neural network output using forward propagation.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        self.base.forward_prop(inputs)
    }

    /// Train the model using gradient optimization and back propagation.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) -> LearningResult<()> {
        let optimal_w = self.alg.optimize(&self.base, &self.base.weights, inputs, targets);
        self.base.weights = optimal_w;
        Ok(())
    }
}

impl<'a> NeuralNet<'a, BCECriterion, StochasticGD> {
    /// Creates a neural network with the specified layer sizes.
    ///
    /// The layer sizes slice should include the input, hidden layers, and output layer sizes.
    /// The type of activation function must be specified.
    ///
    /// Uses the default settings (stochastic gradient descent and sigmoid activation function).
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::NeuralNet;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut net = NeuralNet::default(layers);
    /// ```
    pub fn default(layer_sizes: &[usize]) -> NeuralNet<BCECriterion, StochasticGD> {
        NeuralNet {
            base: BaseNeuralNet::default(layer_sizes),
            alg: StochasticGD::default(),
        }
    }
}

impl<'a, T, A> NeuralNet<'a, T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<'a, T>>
{
    /// Create a new neural network with the specified layer sizes.
    ///
    /// The layer sizes slice should include the input, hidden layers, and output layer sizes.
    /// The type of activation function must be specified.
    ///
    /// Currently defaults to simple batch Gradient Descent for optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::BCECriterion;
    /// use rusty_machine::learning::nnet::NeuralNet;
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut net = NeuralNet::new(layers, BCECriterion::default(), StochasticGD::default());
    /// ```
    pub fn new(layer_sizes: &'a [usize], criterion: T, alg: A) -> NeuralNet<'a, T, A> {
        NeuralNet {
            base: BaseNeuralNet::new(layer_sizes, criterion),
            alg: alg,
        }
    }

    /// Gets matrix of weights between specified layer and forward layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::BaseMatrix;
    /// use rusty_machine::learning::nnet::NeuralNet;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut net = NeuralNet::default(layers);
    ///
    /// let w = &net.get_net_weights(2);
    ///
    /// // We add a bias term to the weight matrix
    /// assert_eq!(w.rows(), 4);
    /// assert_eq!(w.cols(), 3);
    /// ```
    pub fn get_net_weights(&self, idx: usize) -> MatrixSlice<f64> {
        self.base.get_layer_weights(&self.base.weights[..], idx)
    }
}

/// Base Neural Network struct
///
/// This struct cannot be instantianated and is used internally only.
#[derive(Debug)]
pub struct BaseNeuralNet<'a, T: Criterion> {
    layer_sizes: &'a [usize],
    weights: Vec<f64>,
    criterion: T,
}


impl<'a> BaseNeuralNet<'a, BCECriterion> {
    /// Creates a base neural network with the specified layer sizes.
    fn default(layer_sizes: &[usize]) -> BaseNeuralNet<BCECriterion> {
        BaseNeuralNet::new(layer_sizes, BCECriterion::default())
    }
}


impl<'a, T: Criterion> BaseNeuralNet<'a, T> {
    /// Create a new base neural network with the specified layer sizes.
    fn new(layer_sizes: &[usize], criterion: T) -> BaseNeuralNet<T> {
        BaseNeuralNet {
            layer_sizes: layer_sizes,
            weights: BaseNeuralNet::<T>::create_weights(layer_sizes),
            criterion: criterion,
        }
    }

    /// Creates initial weights for all neurons in the network.
    fn create_weights(layer_sizes: &[usize]) -> Vec<f64> {
        let mut between = range::Range::new(0f64, 1f64);
        let mut rng = thread_rng();
        layer_sizes.windows(2)
            .flat_map(|w| {
                let l_in = w[0] + 1;
                let l_out = w[1];
                let eps_init = (6f64 / (l_in + l_out) as f64).sqrt();
                (0..l_in * l_out)
                    .map(|_i| (between.sample(&mut rng) * 2f64 * eps_init) - eps_init)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Gets matrix of weights between specified layer and forward layer for the weights.
    fn get_layer_weights(&self, weights: &[f64], idx: usize) -> MatrixSlice<f64> {
        debug_assert!(idx < self.layer_sizes.len() - 1);

        // Check that the weights are the right size.
        let mut full_size = 0usize;
        for l in 0..self.layer_sizes.len() - 1 {
            full_size += (self.layer_sizes[l] + 1) * self.layer_sizes[l + 1];
        }

        debug_assert_eq!(full_size, weights.len());

        let mut start = 0usize;

        for l in 0..idx {
            start += (self.layer_sizes[l] + 1) * self.layer_sizes[l + 1]
        }

        unsafe {
            MatrixSlice::from_raw_parts(weights.as_ptr().offset(start as isize),
                                        self.layer_sizes[idx] + 1,
                                        self.layer_sizes[idx + 1],
                                        self.layer_sizes[idx + 1])
        }

    }

    /// Gets matrix of weights between specified layer and forward layer
    /// for the base model.
    fn get_net_weights(&self, idx: usize) -> MatrixSlice<f64> {
        self.get_layer_weights(&self.weights[..], idx)
    }

    /// Gets the weights for a layer excluding the bias weights.
    fn get_non_bias_weights(&self, weights: &[f64], idx: usize) -> MatrixSlice<f64> {
        let layer_weights = self.get_layer_weights(weights, idx);
        layer_weights.sub_slice([1, 0], layer_weights.rows() - 1, layer_weights.cols())
    }

    /// Compute the gradient using the back propagation algorithm.
    fn compute_grad(&self,
                    weights: &[f64],
                    inputs: &Matrix<f64>,
                    targets: &Matrix<f64>)
                    -> (f64, Vec<f64>) {
        assert_eq!(inputs.cols(), self.layer_sizes[0]);

        let mut forward_weights = Vec::with_capacity(self.layer_sizes.len() - 1);
        let mut activations = Vec::with_capacity(self.layer_sizes.len());

        let net_data = Matrix::ones(inputs.rows(), 1).hcat(inputs);

        activations.push(net_data.clone());

        // Forward propagation
        {
            let mut z = net_data * self.get_layer_weights(weights, 0);
            forward_weights.push(z.clone());

            for l in 1..self.layer_sizes.len() - 1 {
                let mut a = self.criterion.activate(z.clone());
                let ones = Matrix::ones(a.rows(), 1);

                a = ones.hcat(&a);

                z = &a * self.get_layer_weights(weights, l);
                activations.push(a);
                forward_weights.push(z.clone());
            }

            activations.push(self.criterion.activate(z));
        }

        let mut deltas = Vec::with_capacity(self.layer_sizes.len() - 1);
        // Backward propagation
        {
            let z = forward_weights[self.layer_sizes.len() - 2].clone();
            let g = self.criterion.grad_activ(z);

            // Take GRAD_cost to compute this delta.
            let mut delta = self.criterion
                .cost_grad(&activations[self.layer_sizes.len() - 1], targets)
                .elemul(&g);

            deltas.push(delta.clone());

            for l in (1..self.layer_sizes.len() - 1).rev() {
                let mut z = forward_weights[l - 1].clone();
                let ones = Matrix::ones(z.rows(), 1);
                z = ones.hcat(&z);

                let g = self.criterion.grad_activ(z);
                delta = (delta * Matrix::from(self.get_layer_weights(weights, l)).transpose())
                    .elemul(&g);

                let non_one_rows = &(1..delta.cols()).collect::<Vec<usize>>()[..];
                delta = delta.select_cols(non_one_rows);
                deltas.push(delta.clone());
            }
        }

        let mut gradients = Vec::with_capacity(weights.len());

        for (l, activ_item) in activations.iter().take(self.layer_sizes.len() - 1).enumerate() {
            // Compute the gradient
            let mut g = deltas[self.layer_sizes.len() - 2 - l].transpose() * activ_item;

            // Add the regularized gradient
            if self.criterion.is_regularized() {
                let layer = l;
                let non_bias_weights = self.get_non_bias_weights(weights, layer);
                let zeros = Matrix::zeros(1, non_bias_weights.cols());
                g += zeros.vcat(&self.criterion.reg_cost_grad(non_bias_weights));
            }

            gradients.append(&mut (g / inputs.rows() as f64).into_vec());
        }

        // Compute the cost
        let mut cost = self.criterion.cost(&activations[activations.len() - 1], targets);

        // Add the regularized cost
        if self.criterion.is_regularized() {
            for i in 0..self.layer_sizes.len() - 1 {
                cost += self.criterion.reg_cost(self.get_non_bias_weights(weights, i));
            }
        }

        (cost, gradients)
    }

    /// Forward propagation of the model weights to get the outputs.
    fn forward_prop(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        if inputs.cols() != self.layer_sizes[0] {
            Err(Error::new(ErrorKind::InvalidData,
                           "The input data dimensions must match the first layer."))
        } else {
            let net_data = Matrix::ones(inputs.rows(), 1).hcat(inputs);

            let mut z = net_data * self.get_net_weights(0);
            let mut a = self.criterion.activate(z.clone());

            for l in 1..self.layer_sizes.len() - 1 {
                let ones = Matrix::ones(a.rows(), 1);
                a = ones.hcat(&a);
                z = a * self.get_net_weights(l);
                a = self.criterion.activate(z.clone());
            }

            Ok(a)
        }
    }
}

/// Compute the gradient of the Neural Network using the
/// back propagation algorithm.
impl<'a, T: Criterion> Optimizable for BaseNeuralNet<'a, T> {
    type Inputs = Matrix<f64>;
    type Targets = Matrix<f64>;

    /// Compute the gradient of the neural network.
    fn compute_grad(&self,
                    params: &[f64],
                    inputs: &Matrix<f64>,
                    targets: &Matrix<f64>)
                    -> (f64, Vec<f64>) {
        self.compute_grad(params, inputs, targets)
    }
}

/// Criterion for Neural Networks
///
/// Specifies an activation function and a cost function.
pub trait Criterion {
    /// The activation function for the criterion.
    type ActFunc: ActivationFunc;
    /// The cost function for the criterion.
    type Cost: CostFunc<Matrix<f64>>;

    /// The activation function applied to a matrix.
    fn activate(&self, mat: Matrix<f64>) -> Matrix<f64> {
        mat.apply(&Self::ActFunc::func)
    }

    /// The gradient of the activation function applied to a matrix.
    fn grad_activ(&self, mat: Matrix<f64>) -> Matrix<f64> {
        mat.apply(&Self::ActFunc::func_grad)
    }

    /// The cost function.
    ///
    /// Returns a scalar cost.
    fn cost(&self, outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
        Self::Cost::cost(outputs, targets)
    }

    /// The gradient of the cost function.
    ///
    /// Returns a matrix of cost gradients.
    fn cost_grad(&self, outputs: &Matrix<f64>, targets: &Matrix<f64>) -> Matrix<f64> {
        Self::Cost::grad_cost(outputs, targets)
    }

    /// Returns the regularization for this criterion.
    ///
    /// Will return `Regularization::None` by default.
    fn regularization(&self) -> Regularization<f64> {
        Regularization::None
    }

    /// Checks if the current criterion includes regularization.
    ///
    /// Will return `false` by default.
    fn is_regularized(&self) -> bool {
        match self.regularization() {
            Regularization::None => false,
            _ => true,
        }
    }

    /// Returns the regularization cost for the criterion.
    ///
    /// Will return `0` by default.
    ///
    /// This method will not be invoked by the neural network
    /// if there is explicitly no regularization.
    fn reg_cost(&self, reg_weights: MatrixSlice<f64>) -> f64 {
        self.regularization().reg_cost(reg_weights)
    }

    /// Returns the regularization gradient for the criterion.
    ///
    /// Will return a matrix of zeros by default.
    ///
    /// This method will not be invoked by the neural network
    /// if there is explicitly no regularization.
    fn reg_cost_grad(&self, reg_weights: MatrixSlice<f64>) -> Matrix<f64> {
        self.regularization().reg_grad(reg_weights)
    }
}

/// The binary cross entropy criterion.
///
/// Uses the Sigmoid activation function and the
/// cross entropy error.
#[derive(Clone, Copy, Debug)]
pub struct BCECriterion {
    regularization: Regularization<f64>,
}

impl Criterion for BCECriterion {
    type ActFunc = activ_fn::Sigmoid;
    type Cost = cost_fn::CrossEntropyError;

    fn regularization(&self) -> Regularization<f64> {
        self.regularization
    }
}

/// Creates an MSE Criterion without any regularization.
impl Default for BCECriterion {
    fn default() -> Self {
        BCECriterion { regularization: Regularization::None }
    }
}

impl BCECriterion {
    /// Constructs a new BCECriterion with the given regularization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::BCECriterion;
    /// use rusty_machine::learning::toolkit::regularization::Regularization;
    ///
    /// // Create a new BCE criterion with L2 regularization of 0.3.
    /// let criterion = BCECriterion::new(Regularization::L2(0.3f64));
    /// ```
    pub fn new(regularization: Regularization<f64>) -> Self {
        BCECriterion { regularization: regularization }
    }
}

/// The mean squared error criterion.
///
/// Uses the Linear activation function and the
/// mean squared error.
#[derive(Clone, Copy, Debug)]
pub struct MSECriterion {
    regularization: Regularization<f64>,
}

impl Criterion for MSECriterion {
    type ActFunc = activ_fn::Linear;
    type Cost = cost_fn::MeanSqError;

    fn regularization(&self) -> Regularization<f64> {
        self.regularization
    }
}

/// Creates an MSE Criterion without any regularization.
impl Default for MSECriterion {
    fn default() -> Self {
        MSECriterion { regularization: Regularization::None }
    }
}

impl MSECriterion {
    /// Constructs a new BCECriterion with the given regularization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::MSECriterion;
    /// use rusty_machine::learning::toolkit::regularization::Regularization;
    ///
    /// // Create a new MSE criterion with L2 regularization of 0.3.
    /// let criterion = MSECriterion::new(Regularization::L2(0.3f64));
    /// ```
    pub fn new(regularization: Regularization<f64>) -> Self {
        MSECriterion { regularization: regularization }
    }
}
