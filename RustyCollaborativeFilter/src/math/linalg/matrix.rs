use std::ops::Mul;

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

	fn mul(self, f: f32) -> Matrix {
		
	}
}