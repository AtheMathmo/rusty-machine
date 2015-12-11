use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero, Float};
use std::cmp::PartialEq;
use math::linalg::Metric;
use math::linalg::vector::Vector;
use math::utils::{dot, argmax, find};

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

impl<T: Copy + One + Zero + Neg<Output=T> + Add<T, Output=T>
        + Mul<T, Output=T> + Sub<T, Output=T>
        + Div<T, Output=T> + PartialOrd> Matrix<T> {

    fn solve_u_triangular(&self, y: Vector<T>) -> Vector<T> {
        assert!(self.cols == y.size);

        let mut x = vec![T::zero(); y.size];

        let mut holding_u_sum = T::zero();
        x[y.size-1] = y[y.size-1] / self[[y.size-1,y.size-1]];

        for i in (0..y.size-1).rev() {
            holding_u_sum = holding_u_sum + self[[i,i+1]];
            x[i] = (y[i] - holding_u_sum*x[i+1]) / self[[i,i]];
        }

        Vector {
            size: y.size,
            data: x
        }
    }

    fn solve_l_triangular(&self, y: Vector<T>) -> Vector<T> {
        assert!(self.cols == y.size);

        let mut x = vec![T::zero(); y.size];

        let mut holding_l_sum = T::zero();
        x[0] = y[0] / self[[0,0]];

        for i in 1..y.size {
            holding_l_sum = holding_l_sum + self[[i,i-1]];
            x[i] = (y[i] - holding_l_sum*x[i-1]) / self[[i,i]];
        }

        Vector {
            size: y.size,
            data: x
        }
    }

    pub fn solve(&self, y: Vector<T>) -> Vector<T> {
        let (l,u,p) = self.lup_decomp();

        let b = l.solve_l_triangular(p * y);
        u.solve_u_triangular(b)
    }

    pub fn inverse(&self) -> Matrix<T> {
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

        let sgn = p.parity();

        return sgn * d;
    }

    fn parity(&self) -> T {
        let mut visited = vec![false; self.rows];
        let mut sgn = T::one();

        for k in 0..self.rows {
            if !visited[k] {
                let mut next = k;
                let mut len = 0;

                while !visited[next] {
                    len += 1;
                    visited[next] = true;
                    next = find(&self.data[next*self.cols..(next+1)*self.cols], T::one());
                }

                if len % 2 == 0 {
                    sgn = -sgn;
                }
            }
        }
        sgn
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

impl <T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

impl <'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

impl <'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output=T>> Mul<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        let new_data : Vec<T> = self.data.iter().map(|v| (*v) * (*f)).collect();

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

impl <T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        (&self) * (&m)
    }
}

impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        self * (&m)
    }
}

impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'a Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: &Vector<T>) -> Vector<T> {
        (&self) * m
    }
}

impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, v: &Vector<T>) -> Vector<T> {
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
        (&self) + (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        (&self) + f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Add<T, Output=T>> Add<&'b T> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn add(self, f: &T) -> Matrix<T> {
		let new_data : Vec<T> = self.data.iter().map(|v| (*v) + (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Add<T, Output=T>> Add<&'b Matrix<T>> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn add(self, m: &Matrix<T>) -> Matrix<T> {
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = self.data.iter().enumerate().map(|(i,v)| *v + m.data[i]).collect();
        //let new_data = unrolled_sum(&self.data, &m.data);

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
        (&self) - (&f)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| *v - *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'b Matrix<T>> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn sub(self, m: &Matrix<T>) -> Matrix<T> {
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = self.data.iter().enumerate().map(|(i,v)| *v - m.data[i]).collect();

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
        (&self) / (&f)
    }
}

impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<&'b T> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn div(self, f: &T) -> Matrix<T> {
		assert!(*f != T::zero());
		
		let new_data = self.data.iter().map(|v| *v / *f).collect();

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