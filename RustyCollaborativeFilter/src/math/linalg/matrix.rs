use std::ops::{Mul, Add, Div, Sub, Index};
use math::linalg::vector::Vector;

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

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, v: Vector) -> Vector {
        let mut new_data = vec![0.0; v.size];

        for i in 0..self.rows
        {
            let mut sum = 0.0;
            for j in 0..self.cols
            {
                sum += self.data[i * self.cols + j] * v.data[j];
            }

            new_data[i] = sum;
        }

        return Vector {
            size: v.size,
            data: new_data
        }
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

impl Sub<f32> for Matrix {
	type Output = Matrix;

	fn sub(self, f: f32) -> Matrix {
		let new_data = self.data.into_iter().map(|v| v - f).collect();

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
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = self.data.into_iter().enumerate().map(|(i,v)| v + m.data[i]).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

impl Sub<Matrix> for Matrix {
	type Output = Matrix;

	fn sub(self, m: Matrix) -> Matrix {
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = self.data.into_iter().enumerate().map(|(i,v)| v - m.data[i]).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

impl Div<f32> for Matrix {
	type Output = Matrix;

	fn div(self, f: f32) -> Matrix {
		assert!(f != 0.0);
		
		let new_data = self.data.into_iter().map(|v| v / f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

impl Index<[usize; 2]> for Matrix {
	type Output = f32;

	fn index(&self, idx : [usize; 2]) -> &f32 {
		assert!(idx[0] < self.cols);
		assert!(idx[1] < self.rows);

		&self.data[idx[0] * self.cols + idx[1]]
	}
}