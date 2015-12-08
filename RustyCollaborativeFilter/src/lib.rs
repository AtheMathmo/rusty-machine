extern crate num as libnum;

pub mod math {
	
	pub mod linalg {

		pub trait HasMetric {
			fn norm(&self) -> f32;
		}

		pub mod matrix;
		pub mod vector;
	}

	pub mod utils;
}

pub mod optim {

}