extern crate num as libnum;

pub mod math {
	
	pub mod linalg {

		pub trait Metric<T> {
			fn norm(&self) -> T;
		}

		pub mod matrix;
		pub mod vector;
	}

	pub mod utils;
}

pub mod optim {

}