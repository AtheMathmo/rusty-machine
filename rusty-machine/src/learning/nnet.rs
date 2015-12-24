use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::SupModel;
use rand::{Rng, thread_rng};

/// Neural Network struct
///
/// Requires the number of neurons in each layer to be specified.
pub struct NeuralNet<'a> {
    layer_sizes: &'a [usize],
    weights: Vec<f64>,
}

impl<'a> NeuralNet<'a> {

	/// Create a new neural network with the specified layer sizes.
	///
	/// The layer sizes slice should include the input, hidden layers, and output layer sizes.
	///
	/// # Examples
	///
	/// ```
	/// use rusty_machine::learning::nnet::NeuralNet;
	///
	/// // Create a neural net with 4 layers, 3 neurons in each.
	/// let layers = &[3; 4];
	/// let mut a = NeuralNet::new(layers);
	/// ```
    pub fn new(layer_sizes: &[usize]) -> NeuralNet {
        NeuralNet {
            layer_sizes: layer_sizes,
            weights: NeuralNet::create_weights(layer_sizes),
        }
    }

    /// Creates initial weights for all neurons in the network.
    fn create_weights(layer_sizes: &[usize]) -> Vec<f64> {
        let total_layers = layer_sizes.len();

        let mut capacity = 0usize;

        for l in 0..total_layers - 1 {
            capacity += (layer_sizes[l]+1) * layer_sizes[l + 1]
        }

        let mut layers = Vec::with_capacity(capacity);

        for l in 0..total_layers - 1 {
            layers.append(&mut NeuralNet::initialize_weights(layer_sizes[l]+1, layer_sizes[l + 1]));
        }

        layers
    }

    /// Initializes the weights for a single layer in the network.
    fn initialize_weights(rows: usize, cols: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(rows * cols);
        let eps_init = (6f64).sqrt() / ((rows + cols) as f64).sqrt();

        let mut rng = thread_rng();

        for _i in 0..rows * cols {
            let w = (rng.gen_range(0f64, 1f64) * 2f64 * eps_init) - eps_init;
            weights.push(w);
        }

        weights
    }
}

impl<'a> SupModel<Matrix<f64>, Vector<usize>> for NeuralNet<'a> {
    fn predict(&self, data: Matrix<f64>) -> Vector<usize> {
        unimplemented!()
    }

    fn train(&mut self, data: Matrix<f64>, values: Vector<usize>) {
        unimplemented!()
    }
}
