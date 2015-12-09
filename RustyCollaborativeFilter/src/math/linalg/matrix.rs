use std::ops::{Mul, Add, Div, Sub, Index};
use libnum::{One, Zero, Float};
use std::cmp::PartialEq;
use math::linalg::Metric;
use math::linalg::vector::Vector;
use math::utils::{dot, argmax};

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

impl<T: Copy + Zero + PartialEq> Matrix<T> {
    pub fn is_diag(&self) -> bool {

        for i in 0..self.rows {
            for j in 0..self.cols {
                if (i != j) && (self[[i,j]] != T::zero()) {
                    return false;
                }
            }
        }

        return true;
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>
        + Mul<T, Output=T> + Sub<T, Output=T>
        + Div<T, Output=T> + PartialOrd> Matrix<T> {

    pub fn inverse(&self) -> Matrix<T> {
        let (l,u,p) = self.lup_decomp();

        unimplemented!();
    }

    pub fn det(&self) -> T {
        assert_eq!(self.rows, self.cols);

        let n = self.cols;

        if self.is_diag() {
            let mut d = T::one();

            for i in 0..n {
                d = d * self[[i,i]];
            }

            return d;
        }

        if n == 2 {
            return (self[[0,0]] * self[[1,1]]) - (self[[0,1]] * self[[1,0]]);
        }

        if n == 3 {
            return (self[[0,0]] * self[[1,1]] * self[[2,2]]) + (self[[0,1]] * self[[1,2]] * self[[2,0]])
                    + (self[[0,2]] * self[[1,0]] * self[[2,1]]) - (self[[0,0]] * self[[1,2]] * self[[2,1]])
                    - (self[[0,1]] * self[[1,0]] * self[[2,2]]) - (self[[0,2]] * self[[1,1]] * self[[2,0]]);
        }

        let (l,u,p) = self.lup_decomp();

        let mut d = T::one();

        for i in 0..n {
            d = d * l[[i,i]];
            d = d * u[[i,i]];
        }

        return d;
    }

    pub fn lup_decomp(&self) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        assert!(self.rows == self.cols);

        let n = self.cols;

        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = Matrix::<T>::zeros(n, n);

        let mt = self.transpose();

        let mut p = Matrix::<T>::identity(n);

        // Compute the permutation matrix
        for i in 0..n {
            let row = argmax(&mt.data[i*(n+1)..(i+1)*n]) + i;

            if row != i {
                for j in 0..n {
                    p.data.swap(i*n + j, row*n+j)
                }
            }
        }

        let a_2 = &p * self;

        for i in 0..n {
            l.data[i*(n+1)] = T::one();

            for j in 0..i+1 {
                let mut s1 = T::zero();

                for k in 0..j {
                    s1 = s1 + l.data[j*n + k] * u.data[k*n + i];
                }

                u.data[j*n + i] = a_2[[j,i]] - s1;
            }

            for j in i..n {
                let mut s2 = T::zero();

                for k in 0..i {
                    s2 = s2 + l.data[j*n + k] * u.data[k*n + i];
                }

                l.data[j*n + i] = (a_2[[j,i]] - s2) / u[[i,i]];
            }

        }

        (l,u,p)
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

impl <T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        self * (&m)
    }
}

impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * m
    }
}

impl<'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Matrix<T>> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn mul(self, m: &Matrix<T>) -> Matrix<T> {
		// Will use Strassen algorithm if large, traditional otherwise
		assert!(self.cols == m.rows);

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        let mt = m.transpose();

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

impl<T: Float> Metric<T> for Matrix<T> {
    fn norm(&self) -> T {
        let mut s = T::zero();

        for u in &self.data {
            s = s + (*u) * (*u);
        }

        s.sqrt()
    }
}