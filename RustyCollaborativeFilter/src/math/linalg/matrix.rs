use std::ops::Mul;
use std::ops::Add;
use std::ops::Div;

struct Matrix {
	cols: i32,
	rows: i32,
	data: Vec<f32>
}

impl Matrix {
    fn new(cols: i32, rows: i32, data: Vec<f32>) -> Matrix {
        Matrix {
            cols: cols,
            rows: rows,
            data: data
        }
    }

    fn zeros(cols: i32, rows: i32) -> Matrix {
    	Matrix {
            cols: cols,
            rows: rows,
            data: vec![0.0; (cols*rows) as usize]
        }
    }

    fn identity(size: i32) -> Matrix {
    	let mut data = vec![0.0; (size*size) as usize];

    	for i in 0..size
    	{
    		data[(i*(size+1)) as usize] = 1.0;
    	}

    	Matrix {
            cols: size,
            rows: size,
            data: data
        }
    }
}

impl Mul<f32> for Matrix {
	type Output = Matrix;

	fn mul(self, m: f32) -> Matrix {

		for entry in &self.data {
			entry * m;
		}

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

impl Add<f32> for Matrix {
	type Output = Matrix;

	fn add(self, m: f32) -> Matrix {
		for entry in &self.data {
			entry + m;
		}

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