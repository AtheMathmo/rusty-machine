use linalg::matrix::Matrix;
use rand::{Rng, thread_rng};

pub struct NeuralNet<'a> {
	layer_sizes: &'a [usize]
}

impl<'a> NeuralNet<'a> {
	pub fn new(layer_sizes: &[usize]) -> NeuralNet {
		NeuralNet{layer_sizes: layer_sizes}
	}

	fn create_layers(&self) {
		let total_layers = self.layer_sizes.len();

		let mut layers = Vec::with_capacity(total_layers-1);

		for l in 0..total_layers-1 {
			layers.push(
				initialize_weights(self.layer_sizes[l],self.layer_sizes[l+1]));
		}
	}
}

fn initialize_weights(rows: usize, cols: usize) -> Matrix<f64> {
	let mut weights = Vec::with_capacity(rows * cols);
	let eps_init = (6f64).sqrt() / ((rows + cols) as f64).sqrt();

	let mut rng = thread_rng();

	for _i in 0..rows*cols {
		let w = (rng.gen_range(0f64,1f64) * 2f64 * eps_init) - eps_init;
		weights.push(w);
	}

	Matrix::new(rows, cols, weights)
}