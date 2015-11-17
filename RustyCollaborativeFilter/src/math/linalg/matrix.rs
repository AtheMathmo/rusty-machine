use std::ops::Mul;
use std::ops::Add;
use std::ops::Div;

struct Matrix {
	cols: i32,
	rows: i32
}

impl Mul<f32> for Matrix {
	type Output = Matrix;

	fn mul(self, m: f32) -> Matrix {
		return self;
	}
}

impl Mul<Matrix> for Matrix {
	type Output = Matrix;

	fn mul(self, m: Matrix) -> Matrix {
		// Will use Strassen algorithm if large, traditional otherwise
		return self;
	}
}

impl Add<Matrix> for Matrix {
	type Output = Matrix;

	fn add(self, m: Matrix) -> Matrix {
		return self;
	}
}

impl Div<f32> for Matrix {
	type Output = Matrix;

	fn div(self, f: f32) -> Matrix {
		return self;
	}
}