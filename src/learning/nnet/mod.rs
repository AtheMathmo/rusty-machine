//! Neural Network module
//!
//! Contains implementation of simple feed forward neural network.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
//! use rusty_machine::learning::toolkit::regularization::Regularization;
//! use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
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
//! // We will create a multilayer perceptron and just use the default stochastic gradient descent.
//! let mut model = NeuralNet::mlp(layers, criterion, StochasticGD::default(), Sigmoid);
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
//! The criterions specify a cost function and any regularization.
//!
//! You can define your own criterion by implementing the `Criterion`
//! trait with a concrete `CostFunc`.


pub mod net_layer;

use linalg::{Matrix, MatrixSlice};
use rulinalg::utils;

use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};
use learning::toolkit::activ_fn;
use learning::toolkit::activ_fn::ActivationFunc;
use learning::toolkit::cost_fn;
use learning::toolkit::cost_fn::CostFunc;
use learning::toolkit::regularization::Regularization;
use learning::optim::{Optimizable, OptimAlgorithm};
use learning::optim::grad_desc::StochasticGD;

use self::net_layer::NetLayer;

/// Neural Network Model
///
/// The Neural Network struct specifies a `Criterion` and
/// a gradient descent algorithm.
#[derive(Debug)]
pub struct NeuralNet<T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<T>>
{
    base: BaseNeuralNet<T>,
    alg: A,
}

/// Supervised learning for the Neural Network.
///
/// The model is trained using back propagation.
impl<T, A> SupModel<Matrix<f64>, Matrix<f64>> for NeuralNet<T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<T>>
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

impl NeuralNet<BCECriterion, StochasticGD> {
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
            base: BaseNeuralNet::default(layer_sizes, activ_fn::Sigmoid),
            alg: StochasticGD::default(),
        }
    }
}

impl<T, A> NeuralNet<T, A>
    where T: Criterion,
          A: OptimAlgorithm<BaseNeuralNet<T>>
{
    /// Create a new neural network with no layers
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::BCECriterion;
    /// use rusty_machine::learning::nnet::NeuralNet;
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// // Create a an empty neural net
    /// let mut net = NeuralNet::new(BCECriterion::default(), StochasticGD::default());
    /// ```
    pub fn new(criterion: T, alg: A) -> NeuralNet<T, A> {
        NeuralNet {
            base: BaseNeuralNet::new(criterion),
            alg: alg,
        }
    }

    /// Create a multilayer perceptron with the specified layer sizes.
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
    /// use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut net = NeuralNet::mlp(layers, BCECriterion::default(), StochasticGD::default(), Sigmoid);
    /// ```
    pub fn mlp<U>(layer_sizes: &[usize], criterion: T, alg: A, activ_fn: U) -> NeuralNet<T, A> 
        where U: ActivationFunc + 'static {
        NeuralNet {
            base: BaseNeuralNet::mlp(layer_sizes, criterion, activ_fn),
            alg: alg,
        }
    }

    /// Adds the specified layer to the end of the network
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::BaseMatrix;
    /// use rusty_machine::learning::nnet::BCECriterion;
    /// use rusty_machine::learning::nnet::NeuralNet;
    /// use rusty_machine::learning::nnet::net_layer::Linear;
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// // Create a new neural net 
    /// let mut net = NeuralNet::new(BCECriterion::default(), StochasticGD::default());
    ///
    /// // Give net an input layer of size 3, hidden layer of size 4, and output layer of size 5
    /// // This net will not apply any activation function to the Linear layer outputs
    /// net.add(Box::new(Linear::new(3, 4)))
    ///    .add(Box::new(Linear::new(4, 5)));
    /// ```
    pub fn add<'a>(&'a mut self, layer: Box<dyn NetLayer>) -> &'a mut NeuralNet<T, A> {
        self.base.add(layer);
        self
    }

    /// Adds multiple layers to the end of the network
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::BaseMatrix;
    /// use rusty_machine::learning::nnet::BCECriterion;
    /// use rusty_machine::learning::nnet::NeuralNet;
    /// use rusty_machine::learning::nnet::net_layer::{NetLayer, Linear};
    /// use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
    /// use rusty_machine::learning::optim::grad_desc::StochasticGD;
    ///
    /// // Create a new neural net 
    /// let mut net = NeuralNet::new(BCECriterion::default(), StochasticGD::default());
    ///
    /// let linear_sig: Vec<Box<NetLayer>> = vec![Box::new(Linear::new(5, 5)), Box::new(Sigmoid)];
    ///
    /// // Give net a layer of size 5, followed by a Sigmoid activation function
    /// net.add_layers(linear_sig);
    /// ```
    pub fn add_layers<'a, U>(&'a mut self, layers: U) -> &'a mut NeuralNet<T, A>
        where U: IntoIterator<Item = Box<dyn NetLayer>> {
            self.base.add_layers(layers);
            self
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
/// This struct cannot be instantiated and is used internally only.
#[derive(Debug)]
pub struct BaseNeuralNet<T: Criterion> {
    layers: Vec<Box<dyn NetLayer>>,
    weights: Vec<f64>,
    criterion: T,
}


impl BaseNeuralNet<BCECriterion> {
    /// Creates a base neural network with the specified layer sizes.
    fn default<U>(layer_sizes: &[usize], activ_fn: U) -> BaseNeuralNet<BCECriterion>
        where U: ActivationFunc + 'static {
        BaseNeuralNet::mlp(layer_sizes, BCECriterion::default(), activ_fn)
    }
}


impl<T: Criterion> BaseNeuralNet<T> {
    /// Create a base neural network with no layers
    fn new(criterion: T) -> BaseNeuralNet<T> {
        BaseNeuralNet {
            layers: Vec::new(),
            weights: Vec::new(),
            criterion: criterion
        }
    } 

    /// Create a multilayer perceptron with the specified layer sizes.
    fn mlp<U>(layer_sizes: &[usize], criterion: T, activ_fn: U) -> BaseNeuralNet<T> 
        where U: ActivationFunc + 'static {
        let mut mlp = BaseNeuralNet {
            layers: Vec::with_capacity(2*(layer_sizes.len()-1)),
            weights: Vec::new(),
            criterion: criterion
        };
        for shape in layer_sizes.windows(2) {
            mlp.add(Box::new(net_layer::Linear::new(shape[0], shape[1])));
            mlp.add(Box::new(activ_fn.clone()));
        }
        mlp
    }

    /// Adds the specified layer to the end of the network
    fn add<'a>(&'a mut self, layer: Box<dyn NetLayer>) -> &'a mut BaseNeuralNet<T> {
        self.weights.extend_from_slice(&layer.default_params());
        self.layers.push(layer);
        self
    }

    /// Adds multiple layers to the end of the network
    fn add_layers<'a, U>(&'a mut self, layers: U) -> &'a mut BaseNeuralNet<T>
        where U: IntoIterator<Item = Box<dyn NetLayer>> 
    {
        for layer in layers {
            self.add(layer);
        }
        self
    }

    /// Gets matrix of weights for the specified layer for the weights.
    fn get_layer_weights(&self, weights: &[f64], idx: usize) -> MatrixSlice<f64> {
        debug_assert!(idx < self.layers.len());

        // Check that the weights are the right size.
        let full_size: usize = self.layers.iter().map(|l| l.num_params()).sum();

        debug_assert_eq!(full_size, weights.len());

        let start: usize = self.layers.iter().take(idx).map(|l| l.num_params()).sum();

        let shape = self.layers[idx].param_shape();
        unsafe {
            MatrixSlice::from_raw_parts(weights.as_ptr().offset(start as isize),
                                        shape.0,
                                        shape.1,
                                        shape.1)
        }
    }

    /// Compute the gradient using the back propagation algorithm.
    fn compute_grad(&self,
                    weights: &[f64],
                    inputs: &Matrix<f64>,
                    targets: &Matrix<f64>)
                    -> (f64, Vec<f64>) {
        let mut gradients = Vec::with_capacity(weights.len());
        unsafe {
            gradients.set_len(weights.len());
        }
        // activations[i] is the output of layer[i]
        let mut activations = Vec::with_capacity(self.layers.len());
        // params[i] is the weights for layer[i]
        let mut params = Vec::with_capacity(self.layers.len());

        // Forward propagation
        
        let mut index = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let shape = layer.param_shape();

            let slice = unsafe {
                MatrixSlice::from_raw_parts(weights.as_ptr().offset(index as isize),
                                            shape.0,
                                            shape.1,
                                            shape.1)
            };

            let output = if i == 0 {
                layer.forward(inputs, slice).unwrap()
            } else {
                layer.forward(activations.last().unwrap(), slice).unwrap()
            };

            activations.push(output);
            params.push(slice);
            index += layer.num_params();
        }
        let output = activations.last().unwrap();

        // Backward propagation
        
        // The gradient with respect to the current layer's output
        let mut out_grad = self.criterion.cost_grad(output, targets);
        // at this point index == weights.len()
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let activation = if i == 0 {inputs} else {&activations[i-1]};
            let result = &activations[i];
            index -= layer.num_params();

            let grad_params = &mut gradients[index..index+layer.num_params()];
            grad_params.copy_from_slice(layer.back_params(&out_grad, activation, result, params[i]).data());
            
            out_grad = layer.back_input(&out_grad, activation, result, params[i]);
        }

        let mut cost = self.criterion.cost(output, targets);
        if self.criterion.is_regularized() {
            let all_params = unsafe {
                MatrixSlice::from_raw_parts(weights.as_ptr(), weights.len(), 1, 1)
            };
            utils::in_place_vec_bin_op(&mut gradients,
                                       self.criterion.reg_cost_grad(all_params).data(),
                                       |x, &y| *x = *x + y);
            cost += self.criterion.reg_cost(all_params);
        }
        (cost, gradients)
    }

    /// Forward propagation of the model weights to get the outputs.
    fn forward_prop(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        if self.layers.is_empty() {
            return Ok(inputs.clone());
        }

        let mut outputs = unsafe {
            let shape = self.layers[0].param_shape();
            let slice = MatrixSlice::from_raw_parts(self.weights.as_ptr(),
                                                    shape.0,
                                                    shape.1,
                                                    shape.1);
            self.layers[0].forward(inputs, slice)?
        };

        let mut index = self.layers[0].num_params();
        for layer in self.layers.iter().skip(1) {
            let shape = layer.param_shape();

            let slice = unsafe {
                MatrixSlice::from_raw_parts(self.weights.as_ptr().offset(index as isize),
                                            shape.0,
                                            shape.1,
                                            shape.1)
            };
            
            outputs = match layer.forward(&outputs, slice) {
                Ok(act) => act,
                Err(_) => {return Err(Error::new(ErrorKind::InvalidParameters,
                    "The network's layers do not line up correctly."))}
            };

            index += layer.num_params();
        }
        Ok(outputs)
    }
}

/// Compute the gradient of the Neural Network using the
/// back propagation algorithm.
impl<T: Criterion> Optimizable for BaseNeuralNet<T> {
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
    /// The cost function for the criterion.
    type Cost: CostFunc<Matrix<f64>>;

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
