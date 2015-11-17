use std::ops::Mul;
use std::ops::Add;
use std::ops::Div;

struct Matrix {
	dim1: i32,
	dim2: i32
}

impl Mul<f32> for Matrix {
	type Output = Matrix;

	fn mul(self, f: f32) -> Matrix {

	}
}

impl Mul<Matrix> for Matrix {
	type Output = Matrix;

	fn mul(self, f: Matrix) -> Matrix {
		// Will use Strassen algorithm if large, traditional otherwise
	}
}

impl Add<Matrix> for Matrix {
	type Output = Matrix;

	fn mul(self, f: Matrix) -> Matrix {

	}
}

impl Div<f32> for Matrix {
	type Output = Matrix;

	fn mul(self, f: f32) -> Matrix {

	}
}