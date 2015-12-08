use std::ops::{Mul, Add, Div, Sub, Index};
use libnum::{One, Zero};
use std::cmp::PartialEq;
use math::linalg::HasMetric;
use math::linalg::vector::Vector;
use math::utils::dot;

pub struct Matrix<T> {
	pub cols: usize,
	pub rows: usize,
	pub data: Vec<T>
}

impl<T: Zero + One + Copy> Matrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {

        assert_eq!(cols*rows, data.len());
        Matrix {
            cols: cols,
            rows: rows,
            data: data
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
    	Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::zero(); cols*rows]
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::one(); cols*rows]
        }
    }

    pub fn identity(size: usize) -> Matrix<T> {
    	let mut data = vec![T::zero(); size * size];

    	for i in 0..size
    	{
    		data[(i*(size+1)) as usize] = T::one();
    	}

    	Matrix {
            cols: size,
            rows: size,
            data: data
        }
    }

    pub fn from_diag(diag: &[T]) -> Matrix<T> {
    	let size = diag.len();
    	let mut data = vec![T::zero(); size * size];

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

    pub fn transpose(&self) -> Matrix<T> {
        let mut new_data = vec![T::zero(); self.cols * self.rows];
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

impl<T: Copy + One + Zero + Add<T, Output=T> + Mul<T, Output=T>> Matrix<T> {
    pub fn plu_decomp(&self) -> (Matrix<T>, Matrix<T>) {
        let a = Matrix { cols: self.cols, rows: self.rows, data: vec![T::zero();self.rows*self.cols] };
        let b = Matrix { cols: self.cols, rows: self.rows, data: vec![T::zero();self.rows*self.cols] };

        unimplemented!();
    }
}


impl<T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        let new_data = self.data.into_iter().map(|v| v * f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for Matrix<T> {
	type Output = Matrix<T>;

	fn mul(self, m: Matrix<T>) -> Matrix<T> {
		// Will use Strassen algorithm if large, traditional otherwise
		assert!(self.cols == m.rows);

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        let mt = &m.transpose();

        for i in 0..self.rows
        {
            for j in 0..m.cols
            {
                new_data[i * m.cols + j] = dot( &self.data[(i * self.cols)..((i+1)*self.cols)], &mt.data[(j*m.rows)..((j+1)*m.rows)] );
            }
        }

        Matrix {
            rows: self.rows,
            cols: m.cols,
            data: new_data
        }
	}
}

impl<T: Copy + One + Zero + Mul<T, Output=T> + Add<T, Output=T>> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, v: Vector<T>) -> Vector<T> {
        assert!(v.size == self.cols);

        let mut new_data = vec![T::zero(); self.rows];

        for i in 0..self.rows
        {
            new_data[i] = dot(&self.data[i*self.cols..(i+1)*self.cols], &v.data);
        }

        return Vector {
            size: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>> Add<T> for Matrix<T> {
	type Output = Matrix<T>;

	fn add(self, f: T) -> Matrix<T> {
		let new_data = self.data.into_iter().map(|v| v + f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>> Add<Matrix<T>> for Matrix<T> {
	type Output = Matrix<T>;

	fn add(self, m: Matrix<T>) -> Matrix<T> {
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

impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        let new_data = self.data.into_iter().map(|v| v - f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<Matrix<T>> for Matrix<T> {
	type Output = Matrix<T>;

	fn sub(self, m: Matrix<T>) -> Matrix<T> {
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

impl<T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for Matrix<T> {
	type Output = Matrix<T>;

	fn div(self, f: T) -> Matrix<T> {
		assert!(f != T::zero());
		
		let new_data = self.data.into_iter().map(|v| v / f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

impl<T> Index<[usize; 2]> for Matrix<T> {
	type Output = T;

	fn index(&self, idx : [usize; 2]) -> &T {
		assert!(idx[0] < self.rows);
		assert!(idx[1] < self.cols);

		&self.data[idx[0] * self.cols + idx[1]]
	}
}

impl HasMetric for Matrix<f32> {
    fn norm(&self) -> f32 {
        let mut s = 0.0;

        for u in &self.data {
            s += u*u;
        }

        s.sqrt()
    }
}