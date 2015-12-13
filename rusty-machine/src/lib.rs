extern crate num as libnum;

pub mod math {
	
	pub mod linalg {

		pub trait Metric<T> {
			fn norm(&self) -> T;
		}

		pub mod matrix;
		pub mod vector;
	}

	// this should be private and tested in document.
	pub mod utils;
}