use std::ops::{Mul, Add, Div, Sub, Index};
use math::linalg::vector::Vector;
use math::utils;

pub struct Matrix {
	pub cols: usize,
	pub rows: usize,
	pub data: Vec<f32>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {

        assert_eq!(cols*rows, data.len());
        Matrix {
            cols: cols,
            rows: rows,
            data: data
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
    	Matrix {
            cols: cols,
            rows: rows,
            data: vec![0.0; cols*rows]
        }
    }

    pub fn identity(size: usize) -> Matrix {
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

    pub fn from_diag(diag: &[f32]) -> Matrix {
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

    pub fn transpose(&self) -> Matrix {
        let mut new_data = vec![0.0; self.cols * self.rows];
        for i in 0..self.cols
        {
            for j in 0..self.rows
            {
                new_data[i*self.rows+j] = self.data[j*self.cols + i];
            }
        }

        Matrix {
            cols: self.rows,
            rows: self.cols,
            data: new_data
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
		assert!(self.cols == m.rows);

        let mut new_data = vec![0.0; self.rows * m.cols];

        let mt = &m.transpose();

        for i in 0..self.rows
        {
            for j in 0..m.cols
            {
                new_data[i * m.cols + j] = utils::dot( &self.data[(i * self.cols)..((i+1)*self.cols)], &mt.data[(j*m.rows)..((j+1)*m.rows)] );
            }
        }

        Matrix {
            rows: self.rows,
            cols: m.cols,
            data: new_data
        }
	}
}

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, v: Vector) -> Vector {
        assert!(v.size == self.cols);

        let mut new_data = vec![0.0; self.rows];

        for i in 0..self.rows
        {
            new_data[i] = utils::dot(&self.data[i*self.cols..(i+1)*self.cols], &v.data);
        }

        return Vector {
            size: self.rows,
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
		assert!(idx[0] < self.rows);
		assert!(idx[1] < self.cols);

		&self.data[idx[0] * self.cols + idx[1]]
	}
}