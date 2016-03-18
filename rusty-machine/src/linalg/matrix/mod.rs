//! The matrix module.
//!
//! Currently contains all code
//! relating to the matrix linear algebra struct.

use std::fmt;
use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero, Float, FromPrimitive};
use std::cmp::{PartialEq, min};
use linalg::Metric;
use linalg::vector::Vector;
use linalg::utils;

mod decomposition;

/// The Matrix struct.
///
/// Can be instantiated with any type.
#[derive(Debug)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
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
    ///
    /// assert_eq!(mat.rows(), 2);
    /// assert_eq!(mat.cols(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// - The input data does not match the given dimensions.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {

        assert!(cols * rows == data.len(),
                "Data does not match given dimensions.");
        Matrix {
            cols: cols,
            rows: rows,
            data: data,
        }
    }

    /// Returns the number of rows in the Matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the Matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns a non-mutable reference to the underlying data.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable slice of the underlying data.
    pub fn mut_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the Matrix and returns the Vec of data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone> Clone for Matrix<T> {
    /// Clones the Matrix.
    fn clone(&self) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T: Copy> Matrix<T> {
    /// Select rows from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::<f64>::ones(3,3);
    ///
    /// let b = &a.select_rows(&[2]);
    /// assert_eq!(b.rows(), 1);
    /// assert_eq!(b.cols(), 3);
    ///
    /// let c = &a.select_rows(&[1,2]);
    /// assert_eq!(c.rows(), 2);
    /// assert_eq!(c.cols(), 3);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if row indices exceed the matrix dimensions.
    pub fn select_rows(&self, rows: &[usize]) -> Matrix<T> {

        let mut mat_vec = Vec::with_capacity(rows.len() * self.cols);

        for row in rows {
            assert!(*row < self.rows,
                    "Row index is greater than number of rows.");
        }

        unsafe {
            for row in rows {
                for i in 0..self.cols {
                    mat_vec.push(*self.data.get_unchecked(*row * self.cols + i));
                }
            }
        }
        Matrix {
            cols: self.cols,
            rows: rows.len(),
            data: mat_vec,
        }
    }

    /// Select columns from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::<f64>::ones(3,3);
    /// let b = &a.select_cols(&[2]);
    /// assert_eq!(b.rows(), 3);
    /// assert_eq!(b.cols(), 1);
    ///
    /// let c = &a.select_cols(&[1,2]);
    /// assert_eq!(c.rows(), 3);
    /// assert_eq!(c.cols(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if column indices exceed the matrix dimensions.
    pub fn select_cols(&self, cols: &[usize]) -> Matrix<T> {

        let mut mat_vec = Vec::with_capacity(cols.len() * self.rows);

        for col in cols {
            assert!(*col < self.cols,
                    "Column index is greater than number of columns.");
        }

        unsafe {
            for i in 0..self.rows {
                for col in cols.into_iter() {
                    mat_vec.push(*self.data.get_unchecked(i * self.cols + col));
                }
            }
        }

        Matrix {
            cols: cols.len(),
            rows: self.rows,
            data: mat_vec,
        }
    }

    /// Select block matrix from matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::<f64>::identity(3);
    /// let b = &a.select(&[0,1], &[1,2]);
    ///
    /// // We get the 2x2 block matrix in the upper right corner.
    /// assert_eq!(b.rows(), 2);
    /// assert_eq!(b.cols(), 2);
    ///
    /// // Prints [0,0,1,0]
    /// println!("{:?}", b.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if row or column indices exceed the matrix dimensions.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {

        let mut mat_vec = Vec::with_capacity(cols.len() * rows.len());

        for col in cols {
            assert!(*col < self.cols,
                    "Column index is greater than number of columns.");
        }

        for row in rows {
            assert!(*row < self.rows,
                    "Row index is greater than number of columns.");
        }

        unsafe {
            for row in rows.into_iter() {
                for col in cols.into_iter() {
                    mat_vec.push(*self.data.get_unchecked(row * self.cols + col));
                }
            }
        }

        Matrix {
            cols: cols.len(),
            rows: rows.len(),
            data: mat_vec,
        }
    }

    /// Horizontally concatenates two matrices. With self on the left.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,2, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    /// let b = Matrix::new(3,1, vec![4.0,5.0,6.0]);
    ///
    /// let c = &a.hcat(&b);
    /// assert_eq!(c.cols(), a.cols() + b.cols());
    /// assert_eq!(c[[1, 2]], 5.0);
    /// ```
    ///
    /// # Panics
    ///
    /// - Self and m have different row counts.
    pub fn hcat(&self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.rows == m.rows, "Matrix row counts are not equal.");

        let mut new_data = Vec::with_capacity((self.cols + m.cols) * self.rows);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    new_data.push(*self.data.get_unchecked(i * self.cols + j));
                }

                for j in 0..m.cols {
                    new_data.push(*m.data.get_unchecked(i * m.cols + j));
                }
            }
        }

        Matrix {
            cols: (self.cols + m.cols),
            rows: self.rows,
            data: new_data,
        }
    }

    /// Vertically concatenates two matrices. With self on top.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    /// let b = Matrix::new(1,3, vec![4.0,5.0,6.0]);
    ///
    /// let c = &a.vcat(&b);
    /// assert_eq!(c.rows(), a.rows() + b.rows());
    /// assert_eq!(c[[2, 2]], 6.0);
    /// ```
    ///
    /// # Panics
    ///
    /// - Self and m have different column counts.
    pub fn vcat(&self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Matrix column counts are not equal.");

        let mut new_data = Vec::with_capacity((self.rows + m.rows) * self.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    new_data.push(*self.data.get_unchecked(i * self.cols + j));
                }
            }

            for i in 0..m.rows {
                for j in 0..m.cols {
                    new_data.push(*m.data.get_unchecked(i * m.cols + j));
                }
            }
        }

        Matrix {
            cols: self.cols,
            rows: (self.rows + m.rows),
            data: new_data,
        }
    }

    /// Extract the diagonal of the matrix
    ///
    /// Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::vector::Vector;
    ///
    /// let a = Matrix::new(3,3,vec![1,2,3,4,5,6,7,8,9]);
    /// let b = Matrix::new(3,2,vec![1,2,3,4,5,6]);
    /// let c = Matrix::new(2,3,vec![1,2,3,4,5,6]);
    ///
    /// let d = &a.diag(); // 1,5,9
    /// let e = &b.diag(); // 1,4
    /// let f = &c.diag(); // 1,5
    ///
    /// assert_eq!(*d.data(), vec![1,5,9]);
    /// assert_eq!(*e.data(), vec![1,4]);
    /// assert_eq!(*f.data(), vec![1,5]);
    /// ```
    pub fn diag(&self) -> Vector<T> {
        let mat_min = min(self.rows, self.cols);

        let mut diagonal = Vec::with_capacity(mat_min);
        unsafe {
            for i in 0..mat_min {
                diagonal.push(*self.data.get_unchecked(i * self.cols + i));
            }
        }
        Vector::new(diagonal)
    }

    /// Applies a function to each element in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// fn add_two(a: f64) -> f64 {
    ///     a + 2f64
    /// }
    ///
    /// let a = Matrix::new(2, 2, vec![0.;4]);
    ///
    /// let b = a.apply(&add_two);
    ///
    /// assert_eq!(*b.data(), vec![2.0; 4]);
    /// ```
    pub fn apply(self, f: &Fn(T) -> T) -> Matrix<T> {
        let new_data = self.data.into_iter().map(f).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl<T: Zero + One + Copy> Matrix<T> {
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
            data: vec![T::zero(); cols*rows],
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
            data: vec![T::one(); cols*rows],
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

        for i in 0..size {
            data[(i * (size + 1)) as usize] = T::one();
        }

        Matrix {
            cols: size,
            rows: size,
            data: data,
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

        for (i, item) in diag.into_iter().enumerate().take(size) {
            data[i * (size + 1)] = *item;
        }

        Matrix {
            cols: size,
            rows: size,
            data: data,
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
        for i in 0..self.cols {
            for j in 0..self.rows {
                new_data[i * self.rows + j] = self.data[j * self.cols + i];
            }
        }

        Matrix {
            cols: self.rows,
            rows: self.cols,
            data: new_data,
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

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    if (i != j) && (*self.data.get_unchecked(i * self.cols + j) != T::zero()) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl<T: Copy + Zero + One + Add<T, Output = T>> Matrix<T> {
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
    /// assert_eq!(*c.data(), vec![4.0, 6.0]);
    /// ```
    pub fn sum_rows(&self) -> Vector<T> {
        let mut row_sum = vec![T::zero(); self.cols];

        unsafe {
            for i in 0..self.rows {
                for (j, item) in row_sum.iter_mut().enumerate().take(self.cols) {
                    *item = *item + *self.data.get_unchecked(i * self.cols + j);
                }
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
    /// assert_eq!(*c.data(), vec![3.0, 7.0]);
    /// ```
    pub fn sum_cols(&self) -> Vector<T> {
        let mut col_sum = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            col_sum.push(utils::unrolled_sum(&self.data[i * self.cols..(i + 1) * self.cols]));
        }
        Vector::new(col_sum)
    }

    /// The sum of all elements in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.sum();
    /// assert_eq!(c, 10.0);
    /// ```
    pub fn sum(&self) -> T {
        utils::unrolled_sum(&self.data[..])
    }
}

impl<T: Copy + Zero + Mul<T, Output = T>> Matrix<T> {
    /// The elementwise product of two matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    /// let b = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elemul(&b);
    /// assert_eq!(*c.data(), vec![1.0, 4.0, 9.0, 16.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrices have different row counts.
    /// - The matrices have different column counts.
    pub fn elemul(&self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.rows == m.rows, "Matrix row counts not equal.");
        assert!(self.cols == m.cols, "Matrix column counts not equal.");

        Matrix::new(self.rows, self.cols, utils::ele_mul(&self.data, &m.data))
    }
}

impl<T: Copy + Zero + Div<T, Output = T>> Matrix<T> {
    /// The elementwise division of two matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    /// let b = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elediv(&b);
    /// assert_eq!(*c.data(), vec![1.0; 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrices have different row counts.
    /// - The matrices have different column counts.
    pub fn elediv(&self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.rows == m.rows, "Matrix row counts not equal.");
        assert!(self.cols == m.cols, "Matrix column counts not equal.");

        Matrix::new(self.rows, self.cols, utils::ele_div(&self.data, &m.data))
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
    /// assert_eq!(*c.data(), vec![2.0, 3.0]);
    ///
    /// let d = a.mean(1);
    /// assert_eq!(*d.data(), vec![1.5, 3.5]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The specified axis is not 0 or 1.
    pub fn mean(&self, axis: usize) -> Vector<T> {
        let m: Vector<T>;
        let n: T;
        match axis {
            0 => {
                m = self.sum_rows();
                n = FromPrimitive::from_usize(self.rows).unwrap();
            }
            1 => {
                m = self.sum_cols();
                n = FromPrimitive::from_usize(self.cols).unwrap();
            }
            _ => panic!("Axis must be 0 or 1."),
        }
        m / n
    }

    /// The variance of the matrix along the specified axis.
    ///
    /// Axis 0 - Sample variance of rows.
    /// Axis 1 - Sample variance of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::<f32>::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.variance(0);
    /// assert_eq!(*c.data(), vec![2.0, 2.0]);
    ///
    /// let d = a.variance(1);
    /// assert_eq!(*d.data(), vec![0.5, 0.5]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The specified axis is not 0 or 1.
    pub fn variance(&self, axis: usize) -> Vector<T> {
        let mean = self.mean(axis);

        let n: usize;
        let m: usize;

        match axis {
            0 => {
                n = self.rows;
                m = self.cols;
            }
            1 => {
                n = self.cols;
                m = self.rows;
            }
            _ => panic!("Axis must be 0 or 1."),
        }

        let mut variance = Vector::new(vec![T::zero(); m]);

        for i in 0..n {
            let mut t = Vec::<T>::with_capacity(m);

            unsafe {
                for j in 0..m {
                    match axis {
                        0 => t.push(*self.data.get_unchecked(i * m + j)),
                        1 => t.push(*self.data.get_unchecked(j * n + i)),
                        _ => panic!("Axis must be 0 or 1."),
                    }

                }
            }

            let v = Vector::new(t);

            variance = variance + &(&v - &mean).elemul(&(&v - &mean));
        }

        let var_size: T = FromPrimitive::from_usize(n - 1).unwrap();
        variance / var_size
    }
}

impl<T> Matrix<T> where T: Copy + One + Zero + Neg<Output=T> +
                           Add<T, Output=T> + Mul<T, Output=T> +
                           Sub<T, Output=T> + Div<T, Output=T> +
                           PartialOrd {

    /// Solves an upper triangular linear system.
    fn solve_u_triangular(&self, y: Vector<T>) -> Vector<T> {
        assert!(self.cols == y.size(), "Matrix and Vector dimensions do not agree.");

        let mut x = vec![T::zero(); y.size()];

        x[y.size()-1] = y[y.size()-1] / self[[y.size()-1,y.size()-1]];

        unsafe {
            for i in (0..y.size()-1).rev() {
                let mut holding_u_sum = T::zero();
                for j in (i+1..y.size()).rev() {
                    holding_u_sum = holding_u_sum + *self.data.get_unchecked(i * self.cols + j) * x[j];
                }
                x[i] = (y[i] - holding_u_sum) / *self.data.get_unchecked(i*(self.cols+1));
            }
        }

        Vector::new(x)
    }

    /// Solves a lower triangular linear system.
    fn solve_l_triangular(&self, y: Vector<T>) -> Vector<T> {
        assert!(self.cols == y.size(), "Matrix and Vector dimensions do not agree.");

        let mut x = Vec::with_capacity(y.size());

        x.push(y[0] / self[[0,0]]);

        unsafe {
            for (i,y_item) in y.data().iter().enumerate().take(y.size()).skip(1) {
                let mut holding_l_sum = T::zero();
                for (j, x_item) in x.iter().enumerate().take(i) {
                    holding_l_sum = holding_l_sum + *self.data.get_unchecked(i * self.cols + j) * *x_item;
                }
                x.push((*y_item - holding_l_sum) / *self.data.get_unchecked(i*(self.cols+1)));
            }
        }

        Vector::new(x)
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
    /// assert_eq!(*x.data(), vec![2.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix column count and vector size are different.
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
    /// assert_eq!(*I.data(), vec![1.0,0.0,0.0,1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn inverse(&self) -> Matrix<T> {
        assert!(self.rows==self.cols, "Matrix is not square.");

        let mut new_t_data = Vec::<T>::new();
        let (l,u,p) = self.lup_decomp();

        let mut d = T::one();

        unsafe {
            for i in 0..l.cols {
                d = d * *l.data.get_unchecked(i*(l.cols+1));
                d = d * *u.data.get_unchecked(i*(u.cols+1));
            }
        }

        if d == T::zero() {
            panic!("Matrix has zero determinant.")
        }

        for i in 0..self.rows {
            let mut id_col = vec![T::zero(); self.cols];
            id_col[i] = T::one();

            let b = l.solve_l_triangular(&p * Vector::new(id_col));
            new_t_data.append(&mut u.solve_u_triangular(b).into_vec());

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
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn det(&self) -> T {
        assert!(self.rows==self.cols, "Matrix is not square.");

        let n = self.cols;

        if self.is_diag() {
            let mut d = T::one();

            unsafe {
                for i in 0..n {
                    d = d * *self.data.get_unchecked(i*(self.cols+1));
                }
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

        unsafe {
            for i in 0..l.cols {
                d = d * *l.data.get_unchecked(i*(l.cols+1));
                d = d * *u.data.get_unchecked(i*(u.cols+1));
            }
        }

        let sgn = p.parity();

        sgn * d
    }
}

/// Multiplies matrix by scalar.
impl<T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

/// Multiplies matrix by scalar.
impl<'a, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.data.iter().map(|v| (*v) * (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Multiplies matrix by matrix.
impl<T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<Matrix<T>> for Matrix<T> {
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
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        unsafe {
            for i in 0..self.rows
            {
                for k in 0..m.rows
                {
                    for j in 0..m.cols
                    {
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) + *self.data.get_unchecked(i * self.cols + k) * *m.data.get_unchecked(k*m.cols + j);
                    }
                }
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
impl<T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<Vector<T>> for Matrix<T> {
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
        assert!(v.size() == self.cols, "Matrix and Vector dimensions do not agree.");

        let mut new_data = Vec::with_capacity(self.rows);

        for i in 0..self.rows
        {
            new_data.push(utils::dot(&self.data[i*self.cols..(i+1)*self.cols], v.data()));
        }

        Vector::new(new_data)
    }
}

/// Adds scalar to matrix.
impl<T: Copy + One + Zero + Add<T, Output = T>> Add<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        (&self) + f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.data.iter().map(|v| (*v) + (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix.
impl<T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = utils::vec_sum(&self.data, &m.data);

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts scalar from matrix.
impl<T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts scalar from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| *v - *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts matrix from matrix.
impl<T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = utils::vec_sub(&self.data, &m.data);

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Divides matrix by scalar.
impl<T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        (&self) / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'a T> for Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'b T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        assert!(*f != T::zero());

        let new_data = self.data.iter().map(|v| *v / *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix.
impl<T: Neg<Output = T> + Copy> Neg for Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix.
impl<'a, T: Neg<Output = T> + Copy> Neg for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Indexes matrix.
///
/// Takes row index first then column.
impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");
        unsafe { &self.data.get_unchecked(idx[0] * self.cols + idx[1]) }
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
        let s = utils::dot(&self.data[..], &self.data[..]);

        s.sqrt()
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    /// Formats the Matrix for display.
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut max_datum_width = 0;
        for datum in &self.data {
            let datum_width = match f.precision() {
                Some(places) => format!("{:.1$}", datum, places).len(),
                None => format!("{}", datum).len()
            };
            if datum_width > max_datum_width {
                max_datum_width = datum_width;
            }
        }
        let width = max_datum_width;

        fn write_row<T: fmt::Display>(f: &mut fmt::Formatter, row: &[T],
                                      left_delimiter: &str,
                                      right_delimiter: &str,
                                      width: usize)
                                      -> Result<(), fmt::Error> {
                try!(write!(f, "{}", left_delimiter));
                for (index, datum) in row.iter().enumerate() {
                    match f.precision() {
                        Some(places) => {
                            try!(write!(f, "{:1$.2$}", datum, width, places));
                        },
                        None => {
                            try!(write!(f, "{:1$}", datum, width));
                        }
                    }
                    if index < row.len() - 1 {
                        try!(write!(f, " "));
                    }
                }
            write!(f, "{}", right_delimiter)
        }

        match self.rows {
            1 => write_row(f, &self.data, "[", "]", width),
            _ => {
                try!(write_row(f, &self.data[0..self.cols],
                               "⎡", // \u{23a1} LEFT SQUARE BRACKET UPPER CORNER
                               "⎤", // \u{23a4} RIGHT SQUARE BRACKET UPPER CORNER
                               width));
                try!(f.write_str("\n"));
                for row_index in 1..self.rows-1 {
                    try!(write_row(f,
                                   &self.data[row_index*self.cols..
                                              (row_index+1)*self.cols],
                                   "⎢", // \u{23a2} LEFT SQUARE BRACKET EXTENSION
                                   "⎥", // \u{23a5} RIGHT SQUARE BRACKET EXTENSION
                                   width));
                    try!(f.write_str("\n"));
                }
                write_row(f,
                          &self.data[(self.rows-1)*self.cols..
                                     self.rows*self.cols],
                          "⎣", // \u{23a3} LEFT SQUARE BRACKET LOWER CORNER
                          "⎦", // \u{23a6} RIGHT SQUARE BRACKET LOWER CORNER
                          width)
            }
        }

    }
}


#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_display_formatting() {
        let first_matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let first_expectation = "⎡1 2 3⎤\n\
                                 ⎣4 5 6⎦";
        assert_eq!(first_expectation, format!("{}", first_matrix));

        let second_matrix = Matrix::new(4, 3, vec![3.14, 2.718, 1.414,
                                                   2.503, 4.669, 1.202,
                                                   1.618, 0.5772, 1.3,
                                                   2.68545, 1.282, 10000.]);
        let second_expectation = "⎡   3.14   2.718   1.414⎤\n\
                                  ⎢  2.503   4.669   1.202⎥\n\
                                  ⎢  1.618  0.5772     1.3⎥\n\
                                  ⎣2.68545   1.282   10000⎦";
        assert_eq!(second_expectation, format!("{}", second_matrix));
    }

    #[test]
    fn test_single_row_display_formatting() {
        let one_row_matrix = Matrix::new(1, 4, vec![1, 2, 3, 4]);
        assert_eq!("[1 2 3 4]", format!("{}", one_row_matrix));
    }

    #[test]
    fn test_display_formatting_precision() {
        let our_matrix = Matrix::new(2, 3, vec![1.2, 1.23, 1.234,
                                                1.2345, 1.23456, 1.234567]);
        let expectations = vec![
            "⎡1.2 1.2 1.2⎤\n\
             ⎣1.2 1.2 1.2⎦",

            "⎡1.20 1.23 1.23⎤\n\
             ⎣1.23 1.23 1.23⎦",

            "⎡1.200 1.230 1.234⎤\n\
             ⎣1.234 1.235 1.235⎦",

            "⎡1.2000 1.2300 1.2340⎤\n\
             ⎣1.2345 1.2346 1.2346⎦"
        ];

        for (places, &expectation) in (1..5).zip(expectations.iter()) {
            assert_eq!(expectation, format!("{:.1$}", our_matrix, places));
        }
    }

}
