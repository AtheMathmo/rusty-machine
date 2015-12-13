extern crate num as libnum;

pub mod linalg {

	pub trait Metric<T> {
		fn norm(&self) -> T;
	}

	pub mod matrix;
	pub mod vector;
	pub mod utils;
}

pub mod learning {
	pub mod lin_reg;

	pub trait Model<T,U> {
		fn predict(&self, data:T) -> U;

		fn train(&mut self, data:T, value:U);
	}
}

pub mod stats {
	pub mod dist {
		pub trait Distribution<T> {
			fn pdf(&self, x:T) -> f64;

			fn logpdf(&self, x: T) -> f64;

			fn sample(&self) -> T;
		}
	}
}