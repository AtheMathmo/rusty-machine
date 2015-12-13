use learning::Model;
use linalg::matrix::Matrix;
use linalg::vector::Vector;

pub struct LinRegressor {
	pub b: Option<Vector<f64>>
}

impl LinRegressor {
	pub fn new() -> LinRegressor {
		LinRegressor { b: None }
	}
}

impl Model<Matrix<f64>, Vector<f64>> for LinRegressor {

	fn predict(&self, data: Matrix<f64>) -> Vector<f64> {
		match self.b {
			Some(ref v) => data * v,
			None => panic!("Model has not been trained.")
		}
	}

	fn train(&mut self, data: Matrix<f64>, values: Vector<f64>) {
		let xt = data.transpose();

		self.b = Some(((&xt * data).inverse() * &xt) * values);
	}
}
