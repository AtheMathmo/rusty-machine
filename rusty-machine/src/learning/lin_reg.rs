use learning::Model;
use math::linalg::matrix::Matrix;
use math::linalg::vector::Vector;

pub struct LinRegressor {
	pub b: Vector<f64>
}

impl Model<Matrix<f64>, Vector<f64>> for LinRegressor {

	fn predict(&self, data: Matrix<f64>) -> Vector<f64> {
		data * &self.b
	}

	fn train(&mut self, data: Matrix<f64>, values: Vector<f64>) {
		let xt = data.transpose();

		self.b = ((&xt * data).inverse() * &xt) * values;
	}
}
