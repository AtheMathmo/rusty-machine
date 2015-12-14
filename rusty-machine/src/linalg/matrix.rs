//! The matrix module.
//! 
//! Currently contains all code
//! relating to the matrix linear algebra struct.

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero, Float, FromPrimitive};
use std::cmp::PartialEq;
use linalg::Metric;
use linalg::vector::Vector;
use linalg::utils;

/// The Matrix struct.
///
/// Can be instantiated with any type.
pub struct Matrix<T> {
	pub cols: usize,
	pub rows: usize,
	pub data: Vec<T>
}

impl<T: Zero + One + Copy> Matrix<T> {

    /// Constructor for Matrix struct.
    ///
    /// Requires both the row and column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let mat = Matrix::new(2,2, vec![1.0,2.0,3.0,4.0]);
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {

        assert_eq!(cols*rows, data.len());
        Matrix {
            cols: cols,
            rows: rows,
            data: data
        }
    }

    /// Constructs matrix of all zeros.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let mat = Matrix::<f64>::zeros(2,3);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
    	Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::zero(); cols*rows]
        }
    }

    /// Constructs matrix of all ones.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let mat = Matrix::<f64>::ones(2,3);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            cols: cols,
            rows: rows,
            data: vec![T::one(); cols*rows]
        }
    }

    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let I = Matrix::<f64>::identity(4);
    /// ```
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

    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let mat = Matrix::from_diag(&vec![1.0,2.0,3.0,4.0]);
    /// ```
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

    /// Tranposes the given matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// 
    /// let mat = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    ///
    /// let mt = mat.transpose();
    /// ```
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

impl<T: Copy + Zero + One + PartialEq> Matrix<T> {

    /// Checks if matrix is diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2, vec![1.0,0.0,0.0,1.0]);
    /// let a_diag = a.is_diag();
    ///
    /// assert_eq!(a_diag, true);
    ///
    /// let b = Matrix::new(2,2, vec![1.0,0.0,1.0,0.0]);
    /// let b_diag = b.is_diag();
    ///
    /// assert_eq!(b_diag, false);
    /// ```
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

impl<T: Copy + Zero + One + Add<T, Output=T>> Matrix<T> {

    /// The sum of the rows of the matrix.
    ///
    /// Returns a Vector equal to the sum of the matrices rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.sum_rows();
    /// assert_eq!(c.data, vec![4.0, 6.0]);
    /// ```
    pub fn sum_rows(&self) -> Vector<T> {
        let mut row_sum = vec![T::zero(); self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                row_sum[j] = row_sum[j] + self[[i, j]];
            } 
        }
        Vector::new(row_sum)
    }

    /// The sum of the columns of the matrix.
    ///
    /// Returns a Vector equal to the sum of the matrices columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.sum_cols();
    /// assert_eq!(c.data, vec![3.0, 7.0]);
    /// ```
    pub fn sum_cols(&self) -> Vector<T> {
        let mut col_sum = vec![T::zero(); self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                col_sum[i] = col_sum[i] + self[[i, j]];
            } 
        }
        Vector::new(col_sum)
    }
}

impl<T: Copy + Zero + Float + FromPrimitive> Matrix<T> {

    /// The mean of the matrix along the specified axis.
    ///
    /// Axis 0 - Arithmetic mean of rows.
    /// Axis 1 - Arithmetic mean of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::<f64>::new(2,2, vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.mean(0);
    /// assert_eq!(c.data, vec![2.0, 3.0]);
    /// ```
    pub fn mean(&self, axis: usize) -> Vector<T> {
        let m : Vector<T>;
        let n : T;
        match axis {
            0 => {m = self.sum_rows(); n = FromPrimitive::from_usize(self.rows).unwrap();},
            1 => {m = self.sum_cols(); n = FromPrimitive::from_usize(self.cols).unwrap();},
            _ => panic!("Axis must be 0 or 1."),
        }
        m / n
    }
}

impl<T: Copy + One + Zero + Neg<Output=T> + Add<T, Output=T>
        + Mul<T, Output=T> + Sub<T, Output=T>
        + Div<T, Output=T> + PartialOrd> Matrix<T> {

    /// Solves an upper triangular linear system.
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

    /// Solves a lower triangular linear system.
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

    /// Solves the equation Ax = y.
    ///
    /// Requires a Vector y as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::vector::Vector;
    ///
    /// let a = Matrix::new(2,2, vec![2.0,3.0,1.0,2.0]);
    /// let y = Vector::new(vec![13.0,8.0]);
    ///
    /// let x = a.solve(y);
    ///
    /// assert_eq!(x.data, vec![2.0, 3.0]);
    /// ```
    pub fn solve(&self, y: Vector<T>) -> Vector<T> {
        let (l,u,p) = self.lup_decomp();

        let b = l.solve_l_triangular(p * y);
        u.solve_u_triangular(b)
    }

    /// Computes the inverse of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2, vec![2.,3.,1.,2.]);
    /// let inv = a.inverse();
    ///
    /// let I = a * inv;
    ///
    /// assert_eq!(I.data, vec![1.0,0.0,0.0,1.0]);
    /// ```
    pub fn inverse(&self) -> Matrix<T> {
        assert_eq!(self.rows, self.cols);

        let mut new_t_data = Vec::<T>::new();
        let (l,u,p) = self.lup_decomp();

        let mut d = T::one();

        for i in 0..l.cols {
            d = d * l[[i,i]];
            d = d * u[[i,i]];
        }

        if d == T::zero() {
            panic!("Matrix has zero determinant.")
        }

        for i in 0..self.rows {
            let mut id_col = vec![T::zero(); self.cols];
            id_col[i] = T::one();

            let b = l.solve_l_triangular(&p * Vector::new(id_col));
            new_t_data.append(&mut u.solve_u_triangular(b).data);

        }

        Matrix::new(self.rows, self.cols, new_t_data).transpose()
    }

    /// Computes the determinant of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let det = a.det();
    ///
    /// ```
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

        for i in 0..l.cols {
            d = d * l[[i,i]];
            d = d * u[[i,i]];
        }

        let sgn = p.parity();

        return sgn * d;   
    }

    /// Computes the parity of a permutation matrix.
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
                    next = utils::find(&self.data[next*self.cols..(next+1)*self.cols], T::one());
                }

                if len % 2 == 0 {
                    sgn = -sgn;
                }
            }
        }
        sgn
    }

    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
    ///                               0.0,3.0,4.0,
    ///                               5.0, 1.0, 2.0]);
    ///
    /// let (l,u,p) = a.lup_decomp();
    ///
    /// ``` 
    pub fn lup_decomp(&self) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        assert!(self.rows == self.cols);

        let n = self.cols;

        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = Matrix::<T>::zeros(n, n);

        let mt = self.transpose();

        let mut p = Matrix::<T>::identity(n);

        // Compute the permutation matrix
        for i in 0..n {
            let row = utils::argmax(&mt.data[i*(n+1)..(i+1)*n]) + i;

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

/// Multiplies matrix by scalar.
impl <T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

/// Multiplies matrix by scalar.
impl <'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

/// Multiplies matrix by scalar.
impl <'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

/// Multiplies matrix by scalar.
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

/// Multiplies matrix by matrix.
impl <T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix by matrix.
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

/// Multiplies matrix by vector.
impl <T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by vector.
impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        self * (&m)
    }
}

/// Multiplies matrix by vector.
impl <'a, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'a Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, m: &Vector<T>) -> Vector<T> {
        (&self) * m
    }
}

/// Multiplies matrix by vector.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, v: &Vector<T>) -> Vector<T> {
        assert!(v.size == self.cols);

        let mut new_data = vec![T::zero(); self.rows];

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

/// Adds scalar to matrix.
impl<T: Copy + One + Zero + Add<T, Output=T>> Add<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds scalar to matrix.
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

/// Adds matrix to matrix.
impl<T: Copy + One + Zero + Add<T, Output=T>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output=T>> Add<&'b Matrix<T>> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn add(self, m: &Matrix<T>) -> Matrix<T> {
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = utils::vec_sum(&self.data, &m.data);

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

/// Subtracts scalar from matrix.
impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts scalar from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts scalar from matrix.
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

/// Subtracts matrix from matrix.
impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'b Matrix<T>> for &'a Matrix<T> {
	type Output = Matrix<T>;

	fn sub(self, m: &Matrix<T>) -> Matrix<T> {
		assert!(self.cols == m.cols);
		assert!(self.rows == m.rows);

		let new_data = utils::vec_sub(&self.data, &m.data);

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data
        }
	}
}

/// Divides matrix by scalar.
impl<T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        (&self) / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

/// Divides matrix by scalar.
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

/// Indexes matrix.
///
/// Takes row index first then column.
impl<T> Index<[usize; 2]> for Matrix<T> {
	type Output = T;

	fn index(&self, idx : [usize; 2]) -> &T {
		assert!(idx[0] < self.rows);
		assert!(idx[1] < self.cols);

		&self.data[idx[0] * self.cols + idx[1]]
	}
}

impl<T: Float> Metric<T> for Matrix<T> {
    
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::Metric;
    ///
    /// let a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let c = a.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for u in &self.data {
            s = s + (*u) * (*u);
        }

        s.sqrt()
    }
}