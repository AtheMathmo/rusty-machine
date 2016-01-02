//! Neural Network module
//!
//! Contains implementation of simple feed forward neural network.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::nnet::NeuralNet;
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::learning::SupModel;
//!
//! let data = Matrix::new(5,3, vec![1.,1.,1.,2.,2.,2.,3.,3.,3.,
//! 								4.,4.,4.,5.,5.,5.,]);
//! let outputs = Matrix::new(5,3, vec![1.,0.,0.,0.,1.,0.,0.,0.,1.,
//! 									0.,0.,1.,0.,0.,1.]);
//!
//! let layers = &[3,5,11,7,3];
//! let mut model = NeuralNet::default(layers);
//!
//! model.train(&data, &outputs);
//!
//! let test_data = Matrix::new(2,3, vec![1.5,1.5,1.5,5.1,5.1,5.1]);
//!
//! model.predict(&test_data);
//! ```

use linalg::matrix::Matrix;
use learning::SupModel;
use learning::toolkit::link_fn;
use learning::toolkit::link_fn::LinkFunc;
use learning::optim::{Optimizable, OptimAlgorithm};
use learning::optim::grad_desc::GradientDesc;

use std::marker::PhantomData;
use rand::{Rng, thread_rng};

/// Neural Network struct
pub struct NeuralNet<'a, L: LinkFunc> {
    layer_sizes: &'a [usize],
    weights: Vec<f64>,
    gd: GradientDesc,
    _link: PhantomData<L>,
}

impl<'a> NeuralNet<'a, link_fn::Sigmoid> {

    /// Creates a neural network with the specified layer sizes.
    ///
    /// Uses the default settings (gradient descent and sigmoid link function).
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::NeuralNet;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut a = NeuralNet::default(layers);
    /// ```
    pub fn default(layer_sizes: &[usize]) -> NeuralNet<link_fn::Sigmoid> {
        NeuralNet {
            layer_sizes: layer_sizes,
            weights: NeuralNet::<link_fn::Sigmoid>::create_weights(layer_sizes),
            gd: GradientDesc::default(),
            _link: PhantomData,
        }
    }
}
impl<'a, L: LinkFunc> NeuralNet<'a, L> {
    /// Create a new neural network with the specified layer sizes.
    ///
    /// The layer sizes slice should include the input, hidden layers, and output layer sizes.
    /// The type of link function must be specified.
    ///
    /// Currently defaults to simple batch Gradient Descent for optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::NeuralNet;
    /// use rusty_machine::learning::toolkit::link_fn::Linear;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut a = NeuralNet::<Linear>::new(layers);
    /// ```
    pub fn new(layer_sizes: &[usize]) -> NeuralNet<L> {
        NeuralNet {
            layer_sizes: layer_sizes,
            weights: NeuralNet::<L>::create_weights(layer_sizes),
            gd: GradientDesc::default(),
            _link: PhantomData,
        }
    }

    /// Creates initial weights for all neurons in the network.
    fn create_weights(layer_sizes: &[usize]) -> Vec<f64> {
        let total_layers = layer_sizes.len();

        let mut capacity = 0usize;

        for l in 0..total_layers - 1 {
            capacity += (layer_sizes[l] + 1) * layer_sizes[l + 1]
        }

        let mut layers = Vec::with_capacity(capacity);

        for l in 0..total_layers - 1 {
            layers.append(&mut NeuralNet::<L>::initialize_weights(layer_sizes[l] + 1,
                                                             layer_sizes[l + 1]));
        }

        layers
    }

    /// Initializes the weights for a single layer in the network.
    fn initialize_weights(rows: usize, cols: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(rows * cols);
        let eps_init = (6f64 / (rows + cols) as f64).sqrt();

        let mut rng = thread_rng();

        for _i in 0..rows * cols {
            let w = (rng.gen_range(0f64, 1f64) * 2f64 * eps_init) - eps_init;
            weights.push(w);
        }

        weights
    }

    /// Gets matrix of weights between specified layer and forward layer for the weights.
    fn get_layer_weights(&self, weights: &[f64], idx: usize) -> Matrix<f64> {
        assert!(idx < self.layer_sizes.len() - 1);

        // Check that the weights are the right size.
        let mut full_size = 0usize;
        for l in 0..self.layer_sizes.len() - 1 {
            full_size += (self.layer_sizes[l] + 1) * self.layer_sizes[l + 1];
        }

        assert_eq!(full_size, weights.len());

        let mut start = 0usize;

        for l in 0..idx {
            start += (self.layer_sizes[l] + 1) * self.layer_sizes[l + 1]
        }

        let capacity = (self.layer_sizes[idx] + 1) * self.layer_sizes[idx + 1];

        let mut layer_weights = Vec::with_capacity((self.layer_sizes[idx] + 1) *
                                                   self.layer_sizes[idx + 1]);
        unsafe {
            for i in start..start + capacity {
                layer_weights.push(*weights.get_unchecked(i));
            }
        }

        Matrix::new(self.layer_sizes[idx] + 1,
                    self.layer_sizes[idx + 1],
                    layer_weights)

    }

    /// Gets matrix of weights between specified layer and forward layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::nnet::NeuralNet;
    ///
    /// // Create a neural net with 4 layers, 3 neurons in each.
    /// let layers = &[3; 4];
    /// let mut a = NeuralNet::default(layers);
    ///
    /// let w = &a.get_net_weights(2);
    /// assert_eq!(w.rows(), 4);
    /// assert_eq!(w.cols(), 3);
    /// ```
    pub fn get_net_weights(&self, idx: usize) -> Matrix<f64> {
        self.get_layer_weights(&self.weights[..], idx)
    }

    /// Compute the gradient using the back propagation algorithm.
    fn compute_grad(&self, weights: &[f64], data: &Matrix<f64>, outputs: &Matrix<f64>) -> Vec<f64> {
        assert_eq!(data.cols(), self.layer_sizes[0]);

        let mut forward_weights = Vec::with_capacity(self.layer_sizes.len() - 1);
        let mut activations = Vec::with_capacity(self.layer_sizes.len());

        let net_data = Matrix::ones(data.rows(), 1).hcat(data);


        activations.push(net_data.clone());

        // Forward propagation
        {
            let mut z = net_data * self.get_layer_weights(weights, 0);
            forward_weights.push(z.clone());

            for l in 1..self.layer_sizes.len() - 1 {
                let mut a = z.clone().apply(&L::func);
                let ones = Matrix::ones(a.rows(), 1);

                a = ones.hcat(&a);
                activations.push(a.clone());
                z = a * self.get_layer_weights(weights, l);
                forward_weights.push(z.clone());
            }

            activations.push(z.apply(&L::func));
        }

        let mut deltas = Vec::with_capacity(self.layer_sizes.len() - 1);
        // Backward propagation
        {
            let mut delta = &activations[self.layer_sizes.len() - 1] - outputs;
            deltas.push(delta.clone());

            for l in (1..self.layer_sizes.len() - 1).rev() {
                let mut z = forward_weights[l - 1].clone();
                let ones = Matrix::ones(z.rows(), 1);
                z = ones.hcat(&z);

                let g = z.apply(&L::func_grad);
                delta = (delta * self.get_layer_weights(weights, l).transpose()).elemul(&g);

                let non_one_rows = &(1..delta.cols()).collect::<Vec<usize>>()[..];
                delta = delta.select_cols(non_one_rows);
                deltas.push(delta.clone());
            }
        }

        let mut grad = Vec::with_capacity(self.layer_sizes.len() - 1);
        let mut capacity = 0;

        for l in 0..self.layer_sizes.len() - 1 {
            let g = deltas[self.layer_sizes.len() - 2 - l].transpose() * activations[l].clone();
            capacity += g.cols() * g.rows();
            grad.push(g / (data.rows() as f64));
        }

        let mut gradients = Vec::with_capacity(capacity);

        for g in grad {
            gradients.append(&mut g.data.clone());
        }

        gradients
    }

    /// Forward propagation of the model weights to get the outputs.
    fn forward_prop(&self, data: &Matrix<f64>) -> Matrix<f64> {
        assert_eq!(data.cols(), self.layer_sizes[0]);

        let net_data = Matrix::ones(data.rows(), 1).hcat(data);

        let mut z = net_data * self.get_net_weights(0);
        let mut a = z.clone().apply(&L::func);

        for l in 1..self.layer_sizes.len() - 1 {
            let ones = Matrix::ones(a.rows(), 1);
            a = ones.hcat(&a);
            z = a * self.get_net_weights(l);
            a = z.clone().apply(&L::func);
        }

        a
    }
}

impl<'a, L: LinkFunc> Optimizable for NeuralNet<'a, L> {
    type Data = Matrix<f64>;
	type Target = Matrix<f64>;

    /// Compute the gradient of the neural network.
    fn compute_grad(&self, params: &[f64], data: &Matrix<f64>, target: &Matrix<f64>) -> Vec<f64> {
        self.compute_grad(params, data, target)
    }
}

impl<'a, L: LinkFunc> SupModel<Matrix<f64>, Matrix<f64>> for NeuralNet<'a, L> {
    /// Predict neural network output using forward propagation.
    fn predict(&self, data: &Matrix<f64>) -> Matrix<f64> {
        self.forward_prop(data)
    }

    /// Train the model using gradient optimization and back propagation.
    fn train(&mut self, data: &Matrix<f64>, values: &Matrix<f64>) {
        let start = self.weights.clone();
        let optimal_w = self.gd.optimize(self, &start[..], data, values);
        self.weights = optimal_w;
    }
}