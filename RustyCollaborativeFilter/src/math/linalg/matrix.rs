use std::ops::{Mul, Add, Div, Sub, Index};

struct Matrix {
	cols: usize,
	rows: usize,
	data: Vec<f32>
}

impl Matrix {
    fn new(cols: usize, rows: usize, data: Vec<f32>) -> Matrix {
        Matrix {
            cols: cols,
            rows: rows,
            data: data
        }
    }

    fn zeros(cols: usize, rows: usize) -> Matrix {
    	Matrix {
            cols: cols,
            rows: rows,
            data: vec![0.0; cols*rows]
        }
    }

    fn identity(size: usize) -> Matrix {
    	let mut data = vec![0.0; size * size];

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

    fn from_diag(diag: &Vec<f32>) -> Matrix {
    	let size = diag.len();
    	let mut data = vec![0.0; size * size];

    	for i in 0..size
    	{
    		data[(i*(size+1)) as usize] = diag[i];
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

    fn mul(self, f: f32) -> Matrix {
        let new_data = self.data.into_iter().map(|v| v * f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
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

	fn add(self, f: f32) -> Matrix {
		let new_data = self.data.into_iter().map(|v| v + f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
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
		let new_data = self.data.into_iter().map(|v| v / f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}