//! The matrix module.
//!
//! Currently contains all code
//! relating to the matrix linear algebra struct.

use std::any::Any;
use std::fmt;
use std::ops::{Mul, Add, Div};
use std::cmp::{PartialEq, min};
use std::marker::PhantomData;
use libnum::{One, Zero, Float, FromPrimitive};

use Metric;
use error::{Error, ErrorKind};
use utils;
use vector::Vector;

mod decomposition;
mod impl_ops;
mod mat_mul;
mod iter;
pub mod slice;

/// Matrix dimensions
#[derive(Debug, Clone, Copy)]
pub enum Axes {
    /// The row axis.
    Row,
    /// The column axis.
    Col,
}

/// The `Matrix` struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// A `MatrixSlice`
///
/// This struct provides a slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixSlice<'a, T: 'a> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a T>,
}

/// A mutable `MatrixSliceMut`
///
/// This struct provides a mutable slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug)]
pub struct MatrixSliceMut<'a, T: 'a> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    marker: PhantomData<&'a mut T>,
}

impl<T> Matrix<T> {
    /// Constructor for Matrix struct.
    ///
    /// Requires both the row and column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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
    pub fn new<U: Into<Vec<T>>>(rows: usize, cols: usize, data: U) -> Matrix<T> {
        let our_data = data.into();

        assert!(cols * rows == our_data.len(),
                "Data does not match given dimensions.");
        Matrix {
            cols: cols,
            rows: rows,
            data: our_data,
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

    /// Returns the row-stride of the matrix. This is simply
    /// its column count.
    pub fn row_stride(&self) -> usize {
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

    /// Get a reference to a point in the matrix without bounds checks.
    pub unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        self.data.get_unchecked(index[0] * self.cols + index[1])
    }

    /// Get a mutable reference to a point in the matrix without bounds checks.
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; 2]) -> &T {
        self.data.get_unchecked_mut(index[0] * self.cols + index[1])
    }

    /// Returns pointer to first element of underlying data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Consumes the Matrix and returns the Vec of data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Split the matrix at the specified axis returning two `MatrixSlice`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::Axes;
    ///
    /// let a = Matrix::new(3,3, vec![2.0; 9]);
    /// let (b,c) = a.split_at(1, Axes::Row);
    /// ```
    pub fn split_at(&self, mid: usize, axis: Axes) -> (MatrixSlice<T>, MatrixSlice<T>) {
        let slice_1: MatrixSlice<T>;
        let slice_2: MatrixSlice<T>;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                slice_1 = MatrixSlice::from_matrix(self, [0, 0], mid, self.cols);
                slice_2 = MatrixSlice::from_matrix(self, [mid, 0], self.rows - mid, self.cols);
            }
            Axes::Col => {
                assert!(mid < self.cols);

                slice_1 = MatrixSlice::from_matrix(self, [0, 0], self.rows, mid);
                slice_2 = MatrixSlice::from_matrix(self, [0, mid], self.rows, self.cols - mid);
            }
        }

        (slice_1, slice_2)
    }

    /// Split the matrix at the specified axis returning two `MatrixSlice`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::Axes;
    ///
    /// let mut a = Matrix::new(3,3, vec![2.0; 9]);
    /// let (b,c) = a.split_at_mut(1, Axes::Col);
    /// ```
    pub fn split_at_mut(&mut self,
                        mid: usize,
                        axis: Axes)
                        -> (MatrixSliceMut<T>, MatrixSliceMut<T>) {

        let mat_cols = self.cols;
        let mat_rows = self.rows;

        let slice_1: MatrixSliceMut<T>;
        let slice_2: MatrixSliceMut<T>;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                unsafe {
                    slice_1 = MatrixSliceMut::from_raw_parts(self.data.as_mut_ptr(),
                                                             mid,
                                                             mat_cols,
                                                             mat_cols);
                    slice_2 =
                        MatrixSliceMut::from_raw_parts(self.data
                                                           .as_mut_ptr()
                                                           .offset((mid * mat_cols) as isize),
                                                       mat_rows - mid,
                                                       mat_cols,
                                                       mat_cols);
                }
            }
            Axes::Col => {
                assert!(mid < self.cols);
                unsafe {
                    slice_1 = MatrixSliceMut::from_raw_parts(self.data.as_mut_ptr(),
                                                             mat_rows,
                                                             mid,
                                                             mat_cols);
                    slice_2 = MatrixSliceMut::from_raw_parts(self.data
                                                                 .as_mut_ptr()
                                                                 .offset(mid as isize),
                                                             mat_rows,
                                                             mat_cols - mid,
                                                             mat_cols);
                }
            }
        }

        (slice_1, slice_2)
    }

    /// Returns a `MatrixSlice` over the whole matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3, 3, vec![2.0; 9]);
    /// let b = a.as_slice();
    /// ```
    pub fn as_slice(&self) -> MatrixSlice<T> {
        MatrixSlice::from_matrix(self, [0, 0], self.rows, self.cols)
    }

    /// Returns a mutable `MatrixSlice` over the whole matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mut a = Matrix::new(3, 3, vec![2.0; 9]);
    /// let b = a.as_mut_slice();
    /// ```
    pub fn as_mut_slice(&mut self) -> MatrixSliceMut<T> {
        let rows = self.rows;
        let cols = self.cols;
        MatrixSliceMut::from_matrix(self, [0, 0], rows, cols)
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
    /// use rulinalg::matrix::Matrix;
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

        for row in rows {
            mat_vec.extend_from_slice(&self.data[*row * self.cols..(*row + 1) * self.cols]);
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
    /// use rulinalg::matrix::Matrix;
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
    /// use rulinalg::matrix::Matrix;
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
    /// use rulinalg::matrix::Matrix;
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
    /// use rulinalg::matrix::Matrix;
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
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
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
    /// use rulinalg::matrix::Matrix;
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
    pub fn apply(mut self, f: &Fn(T) -> T) -> Matrix<T> {
        for val in &mut self.data {
            *val = f(*val);
        }

        self
    }

    /// Tranposes the given matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mat = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    ///
    /// let mt = mat.transpose();
    /// ```
    pub fn transpose(&self) -> Matrix<T> {
        let mut new_data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            new_data.set_len(self.rows * self.cols);
        }

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

impl<T: Clone + Zero> Matrix<T> {
    /// Constructs matrix of all zeros.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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

    /// Constructs matrix with given diagonal.
    ///
    /// Requires slice of diagonal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mat = Matrix::from_diag(&vec![1.0,2.0,3.0,4.0]);
    /// ```
    pub fn from_diag(diag: &[T]) -> Matrix<T> {
        let size = diag.len();
        let mut data = vec![T::zero(); size * size];

        for (i, item) in diag.into_iter().enumerate().take(size) {
            data[i * (size + 1)] = item.clone();
        }

        Matrix {
            cols: size,
            rows: size,
            data: data,
        }
    }
}

impl<T: Clone + One> Matrix<T> {
    /// Constructs matrix of all ones.
    ///
    /// Requires both the row and the column dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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
}

impl<T: Clone + Zero + One> Matrix<T> {
    /// Constructs the identity matrix.
    ///
    /// Requires the size of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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
}

impl<T: Copy + Zero + One + PartialEq> Matrix<T> {
    /// Checks if matrix is diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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

impl<T: Copy + Zero + Add<T, Output = T>> Matrix<T> {
    /// The sum of the rows of the matrix.
    ///
    /// Returns a Vector equal to the sum of the matrices rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.sum_cols();
    /// assert_eq!(*c.data(), vec![3.0, 7.0]);
    /// ```
    pub fn sum_cols(&self) -> Vector<T> {
        let mut col_sum = Vec::with_capacity(self.rows);

        for row in self.iter_rows() {
            col_sum.push(utils::unrolled_sum(row));
        }
        Vector::new(col_sum)
    }

    /// The sum of all elements in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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

impl<T: Copy + Mul<T, Output = T>> Matrix<T> {
    /// The elementwise product of two matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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

impl<T: Copy + Div<T, Output = T>> Matrix<T> {
    /// The elementwise division of two matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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

impl<T: Float + FromPrimitive> Matrix<T> {
    /// The mean of the matrix along the specified axis.
    ///
    /// Axis Row - Arithmetic mean of rows.
    /// Axis Col - Arithmetic mean of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, Axes};
    ///
    /// let a = Matrix::<f64>::new(2,2, vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.mean(Axes::Row);
    /// assert_eq!(*c.data(), vec![2.0, 3.0]);
    ///
    /// let d = a.mean(Axes::Col);
    /// assert_eq!(*d.data(), vec![1.5, 3.5]);
    /// ```
    pub fn mean(&self, axis: Axes) -> Vector<T> {
        let m: Vector<T>;
        let n: T;
        match axis {
            Axes::Row => {
                m = self.sum_rows();
                n = FromPrimitive::from_usize(self.rows).unwrap();
            }
            Axes::Col => {
                m = self.sum_cols();
                n = FromPrimitive::from_usize(self.cols).unwrap();
            }
        }
        m / n
    }

    /// The variance of the matrix along the specified axis.
    ///
    /// Axis Row - Sample variance of rows.
    /// Axis Col - Sample variance of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, Axes};
    ///
    /// let a = Matrix::<f32>::new(2,2,vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.variance(Axes::Row);
    /// assert_eq!(*c.data(), vec![2.0, 2.0]);
    ///
    /// let d = a.variance(Axes::Col);
    /// assert_eq!(*d.data(), vec![0.5, 0.5]);
    /// ```
    pub fn variance(&self, axis: Axes) -> Vector<T> {
        let mean = self.mean(axis);

        let n: usize;
        let m: usize;

        match axis {
            Axes::Row => {
                n = self.rows;
                m = self.cols;
            }
            Axes::Col => {
                n = self.cols;
                m = self.rows;
            }
        }

        let mut variance = Vector::zeros(m);

        for i in 0..n {
            let mut t = Vec::<T>::with_capacity(m);

            unsafe {
                t.set_len(m);

                for j in 0..m {
                    t[j] = match axis {
                        Axes::Row => *self.data.get_unchecked(i * m + j),
                        Axes::Col => *self.data.get_unchecked(j * n + i),
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

impl<T: Any + Float> Matrix<T> {
    /// Solves an upper triangular linear system.
    ///
    /// Given a matrix `U`, which is upper triangular, and a vector `y`, this function returns `x`
    /// such that `Ux = y`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use std::f32;
    ///
    /// let u = Matrix::new(2,2, vec![1.0, 2.0, 0.0, 1.0]);
    /// let y = Vector::new(vec![3.0, 1.0]);
    ///
    /// let x = u.solve_u_triangular(y).expect("A solution should exist!");
    /// assert!((x[0] - 1.0) < f32::EPSILON);
    /// assert!((x[1] - 1.0) < f32::EPSILON);
    /// ```
    ///
    /// # Panics
    ///
    /// - Vector size and matrix column count are not equal.
    /// - Matrix is not upper triangular.
    ///
    /// # Failures
    ///
    /// Fails if there is no valid solution to the system (matrix is singular).
    pub fn solve_u_triangular(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        assert!(self.cols == y.size(),
                format!("Vector size {0} != {1} Matrix column count.",
                        y.size(),
                        self.cols));

        // Make sure we are upper triangular.
        for (row_idx, row) in self.iter_rows().enumerate() {
            for i in 0..row_idx {
                unsafe {
                    assert!(*row.get_unchecked(i) == T::zero(),
                            "Matrix is not upper triangular.");
                }

            }
        }

        self.back_substitution(y)
    }

    fn back_substitution(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        let mut x = vec![T::zero(); y.size()];

        x[y.size() - 1] = y[y.size() - 1] / self[[y.size() - 1, y.size() - 1]];

        unsafe {
            for i in (0..y.size() - 1).rev() {
                let mut holding_u_sum = T::zero();
                for j in (i + 1..y.size()).rev() {
                    holding_u_sum = holding_u_sum + *self.get_unchecked([i, j]) * x[j];
                }

                let diag = *self.get_unchecked([i, i]);
                if diag.abs() < T::min_positive_value() + T::min_positive_value() {
                    return Err(Error::new(ErrorKind::AlgebraFailure,
                                          "Linear system cannot be solved (matrix is singular)."));
                }
                x[i] = (y[i] - holding_u_sum) / diag;
            }
        }

        Ok(Vector::new(x))
    }

    /// Solves a lower triangular linear system.
    ///
    /// Given a matrix `L`, which is lower triangular, and a vector `y`, this function returns `x`
    /// such that `Lx = y`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    /// use std::f32;
    ///
    /// let l = Matrix::new(2,2, vec![1.0, 0.0, 2.0, 1.0]);
    /// let y = Vector::new(vec![1.0, 3.0]);
    ///
    /// let x = l.solve_l_triangular(y).expect("A solution should exist!");
    /// println!("{:?}", x);
    /// assert!((x[0] - 1.0) < f32::EPSILON);
    /// assert!((x[1] - 1.0) < f32::EPSILON);
    /// ```
    ///
    /// # Panics
    ///
    /// - Vector size and matrix column count are not equal.
    /// - Matrix is not lower triangular.
    ///
    /// # Failures
    ///
    /// Fails if there is no valid solution to the system (matrix is singular).
    pub fn solve_l_triangular(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        assert!(self.cols == y.size(),
                format!("Vector size {0} != {1} Matrix column count.",
                        y.size(),
                        self.cols));

        // Make sure we are lower triangular.
        for (row_idx, row) in self.iter_rows().enumerate() {
            for i in row_idx + 1..self.cols {
                unsafe {
                    assert!(*row.get_unchecked(i) == T::zero(),
                            "Matrix is not lower triangular.");
                }
            }
        }

        self.forward_substitution(y)
    }

    fn forward_substitution(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        let mut x = Vec::with_capacity(y.size());

        x.push(y[0] / self[[0, 0]]);

        unsafe {
            for (i, y_item) in y.data().iter().enumerate().take(y.size()).skip(1) {
                let mut holding_l_sum = T::zero();
                for (j, x_item) in x.iter().enumerate().take(i) {
                    holding_l_sum = holding_l_sum + *self.get_unchecked([i, j]) * *x_item;
                }

                let diag = *self.get_unchecked([i, i]);

                if diag.abs() < T::min_positive_value() + T::min_positive_value() {
                    return Err(Error::new(ErrorKind::AlgebraFailure,
                                          "Linear system cannot be solved (matrix is singular)."));
                }
                x.push((*y_item - holding_l_sum) / diag);
            }
        }

        Ok(Vector::new(x))
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
                    next = utils::find(&self.data[next * self.cols..(next + 1) * self.cols],
                                       T::one());
                }

                if len % 2 == 0 {
                    sgn = -sgn;
                }
            }
        }
        sgn
    }

    /// Solves the equation `Ax = y`.
    ///
    /// Requires a Vector `y` as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Matrix::new(2,2, vec![2.0,3.0,1.0,2.0]);
    /// let y = Vector::new(vec![13.0,8.0]);
    ///
    /// let x = a.solve(y).unwrap();
    ///
    /// assert_eq!(*x.data(), vec![2.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix column count and vector size are different.
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be decomposed into an LUP form to solve.
    /// - There is no valid solution as the matrix is singular.
    pub fn solve(&self, y: Vector<T>) -> Result<Vector<T>, Error> {
        let (l, u, p) = try!(self.lup_decomp());

        let b = try!(l.forward_substitution(p * y));
        u.back_substitution(b)
    }

    /// Computes the inverse of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(2,2, vec![2.,3.,1.,2.]);
    /// let inv = a.inverse().expect("This matrix should have an inverse!");
    ///
    /// let I = a * inv;
    ///
    /// assert_eq!(*I.data(), vec![1.0,0.0,0.0,1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix could not be LUP decomposed.
    /// - The matrix has zero determinant.
    pub fn inverse(&self) -> Result<Matrix<T>, Error> {
        assert!(self.rows == self.cols, "Matrix is not square.");

        let mut inv_t_data = Vec::<T>::new();
        let (l, u, p) = try!(self.lup_decomp().map_err(|_| {
            Error::new(ErrorKind::DecompFailure,
                       "Could not compute LUP factorization for inverse.")
        }));

        let mut d = T::one();

        unsafe {
            for i in 0..l.cols {
                d = d * *l.get_unchecked([i, i]);
                d = d * *u.get_unchecked([i, i]);
            }
        }

        if d == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Matrix is singular and cannot be inverted."));
        }

        for i in 0..self.rows {
            let mut id_col = vec![T::zero(); self.cols];
            id_col[i] = T::one();

            let b = l.forward_substitution(&p * Vector::new(id_col))
                .expect("Matrix is singular AND has non-zero determinant!?");
            inv_t_data.append(&mut u.back_substitution(b)
                .expect("Matrix is singular AND has non-zero determinant!?")
                .into_vec());

        }

        Ok(Matrix::new(self.rows, self.cols, inv_t_data).transpose())
    }

    /// Computes the determinant of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
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
        assert!(self.rows == self.cols, "Matrix is not square.");

        let n = self.cols;

        if self.is_diag() {
            let mut d = T::one();

            unsafe {
                for i in 0..n {
                    d = d * *self.get_unchecked([i, i]);
                }
            }

            return d;
        }

        if n == 2 {
            (self[[0, 0]] * self[[1, 1]]) - (self[[0, 1]] * self[[1, 0]])
        } else if n == 3 {
            (self[[0, 0]] * self[[1, 1]] * self[[2, 2]]) +
            (self[[0, 1]] * self[[1, 2]] * self[[2, 0]]) +
            (self[[0, 2]] * self[[1, 0]] * self[[2, 1]]) -
            (self[[0, 0]] * self[[1, 2]] * self[[2, 1]]) -
            (self[[0, 1]] * self[[1, 0]] * self[[2, 2]]) -
            (self[[0, 2]] * self[[1, 1]] * self[[2, 0]])
        } else {
            let (l, u, p) = self.lup_decomp().expect("Could not compute LUP decomposition.");

            let mut d = T::one();

            unsafe {
                for i in 0..l.cols {
                    d = d * *l.get_unchecked([i, i]);
                    d = d * *u.get_unchecked([i, i]);
                }
            }

            let sgn = p.parity();

            sgn * d
        }
    }
}

impl<T: Float> Metric<T> for Matrix<T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::Metric;
    ///
    /// let a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let c = a.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let s = utils::dot(&self.data, &self.data);

        s.sqrt()
    }
}

impl<'a, T: Float> Metric<T> for MatrixSlice<'a, T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSlice};
    /// use rulinalg::Metric;
    ///
    /// let a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let b = MatrixSlice::from_matrix(&a, [0,0], 2, 1);
    /// let c = b.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for row in self.iter_rows() {
            s = s + utils::dot(row, row);
        }
        s.sqrt()
    }
}

impl<'a, T: Float> Metric<T> for MatrixSliceMut<'a, T> {
    /// Compute euclidean norm for matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    /// use rulinalg::Metric;
    ///
    /// let mut a = Matrix::new(2,1, vec![3.0,4.0]);
    /// let b = MatrixSliceMut::from_matrix(&mut a, [0,0], 2, 1);
    /// let c = b.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for row in self.iter_rows() {
            s = s + utils::dot(row, row);
        }
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
                None => format!("{}", datum).len(),
            };
            if datum_width > max_datum_width {
                max_datum_width = datum_width;
            }
        }
        let width = max_datum_width;

        fn write_row<T: fmt::Display>(f: &mut fmt::Formatter,
                                      row: &[T],
                                      left_delimiter: &str,
                                      right_delimiter: &str,
                                      width: usize)
                                      -> Result<(), fmt::Error> {
            try!(write!(f, "{}", left_delimiter));
            for (index, datum) in row.iter().enumerate() {
                match f.precision() {
                    Some(places) => {
                        try!(write!(f, "{:1$.2$}", datum, width, places));
                    }
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
                try!(write_row(f,
                               &self.data[0..self.cols],
                               "⎡", // \u{23a1} LEFT SQUARE BRACKET UPPER CORNER
                               "⎤", // \u{23a4} RIGHT SQUARE BRACKET UPPER CORNER
                               width));
                try!(f.write_str("\n"));
                for row_index in 1..self.rows - 1 {
                    try!(write_row(f,
                                   &self.data[row_index * self.cols..(row_index + 1) * self.cols],
                                   "⎢", // \u{23a2} LEFT SQUARE BRACKET EXTENSION
                                   "⎥", // \u{23a5} RIGHT SQUARE BRACKET EXTENSION
                                   width));
                    try!(f.write_str("\n"));
                }
                write_row(f,
                          &self.data[(self.rows - 1) * self.cols..self.rows * self.cols],
                          "⎣", // \u{23a3} LEFT SQUARE BRACKET LOWER CORNER
                          "⎦", // \u{23a6} RIGHT SQUARE BRACKET LOWER CORNER
                          width)
            }
        }

    }
}


#[cfg(test)]
mod tests {
    use super::super::vector::Vector;
    use super::Matrix;
    use super::Axes;
    use super::slice::BaseSlice;
    use libnum::abs;

    #[test]
    fn test_new_mat() {
        let a = vec![2.0; 9];
        let b = Matrix::new(3, 3, a);

        assert_eq!(b.rows(), 3);
        assert_eq!(b.cols(), 3);
        assert_eq!(b.into_vec(), vec![2.0; 9]);
    }

    #[test]
    #[should_panic]
    fn test_new_mat_bad_data() {
        let a = vec![2.0; 7];
        let _ = Matrix::new(3, 3, a);
    }

    #[test]
    fn test_equality() {
        // well, "PartialEq", at least
        let a = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let a_redux = a.clone();
        assert_eq!(a, a_redux);
    }

    #[test]
    fn test_new_from_slice() {
        let data_vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let data_slice: &[u32] = &data_vec[..];
        let from_vec = Matrix::new(3, 2, data_vec.clone());
        let from_slice = Matrix::new(3, 2, data_slice);
        assert_eq!(from_vec, from_slice);
    }

    #[test]
    fn test_display_formatting() {
        let first_matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let first_expectation = "⎡1 2 3⎤\n⎣4 5 6⎦";
        assert_eq!(first_expectation, format!("{}", first_matrix));

        let second_matrix = Matrix::new(4,
                                        3,
                                        vec![3.14, 2.718, 1.414, 2.503, 4.669, 1.202, 1.618,
                                             0.5772, 1.3, 2.68545, 1.282, 10000.]);
        let second_exp = "⎡   3.14   2.718   1.414⎤\n⎢  2.503   4.669   1.202⎥\n⎢  \
                        1.618  0.5772     1.3⎥\n⎣2.68545   1.282   10000⎦";
        assert_eq!(second_exp, format!("{}", second_matrix));
    }

    #[test]
    fn test_single_row_display_formatting() {
        let one_row_matrix = Matrix::new(1, 4, vec![1, 2, 3, 4]);
        assert_eq!("[1 2 3 4]", format!("{}", one_row_matrix));
    }

    #[test]
    fn test_display_formatting_precision() {
        let our_matrix = Matrix::new(2, 3, vec![1.2, 1.23, 1.234, 1.2345, 1.23456, 1.234567]);
        let expectations = vec!["⎡1.2 1.2 1.2⎤\n⎣1.2 1.2 1.2⎦",

                                "⎡1.20 1.23 1.23⎤\n⎣1.23 1.23 1.23⎦",

                                "⎡1.200 1.230 1.234⎤\n⎣1.234 1.235 1.235⎦",

                                "⎡1.2000 1.2300 1.2340⎤\n⎣1.2345 1.2346 1.2346⎦"];

        for (places, &expectation) in (1..5).zip(expectations.iter()) {
            assert_eq!(expectation, format!("{:.1$}", our_matrix, places));
        }
    }

    #[test]
    fn test_split_matrix() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        let (b, c) = a.split_at(1, Axes::Row);

        assert_eq!(b.rows(), 1);
        assert_eq!(b.cols(), 3);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 3);

        assert_eq!(b[[0, 0]], 0);
        assert_eq!(b[[0, 1]], 1);
        assert_eq!(b[[0, 2]], 2);
        assert_eq!(c[[0, 0]], 3);
        assert_eq!(c[[0, 1]], 4);
        assert_eq!(c[[0, 2]], 5);
        assert_eq!(c[[1, 0]], 6);
        assert_eq!(c[[1, 1]], 7);
        assert_eq!(c[[1, 2]], 8);
    }

    #[test]
    fn test_split_matrix_mut() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let (mut b, mut c) = a.split_at_mut(1, Axes::Row);

            assert_eq!(b.rows(), 1);
            assert_eq!(b.cols(), 3);
            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 3);

            assert_eq!(b[[0, 0]], 0);
            assert_eq!(b[[0, 1]], 1);
            assert_eq!(b[[0, 2]], 2);
            assert_eq!(c[[0, 0]], 3);
            assert_eq!(c[[0, 1]], 4);
            assert_eq!(c[[0, 2]], 5);
            assert_eq!(c[[1, 0]], 6);
            assert_eq!(c[[1, 1]], 7);
            assert_eq!(c[[1, 2]], 8);

            b[[0, 0]] = 4;
            c[[0, 0]] = 5;
        }

        assert_eq!(a[[0, 0]], 4);
        assert_eq!(a[[0, 1]], 1);
        assert_eq!(a[[0, 2]], 2);
        assert_eq!(a[[1, 0]], 5);
        assert_eq!(a[[1, 1]], 4);
        assert_eq!(a[[1, 2]], 5);
        assert_eq!(a[[2, 0]], 6);
        assert_eq!(a[[2, 1]], 7);
        assert_eq!(a[[2, 2]], 8);

    }

    #[test]
    fn test_matrix_index_mut() {
        let mut a = Matrix::new(3, 3, vec![2.0; 9]);

        a[[0, 0]] = 13.0;

        for i in 1..9 {
            assert_eq!(a.data()[i], 2.0);
        }

        assert_eq!(a[[0, 0]], 13.0);
    }

    #[test]
    fn test_matrix_select_rows() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select_rows(&[0, 2, 3]);

        assert_eq!(b.into_vec(), vec![0, 1, 4, 5, 6, 7]);
    }

    #[test]
    fn test_matrix_select_cols() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select_cols(&[1]);

        assert_eq!(b.into_vec(), vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_matrix_select() {
        let a = Matrix::new(4, 2, (0..8).collect::<Vec<usize>>());

        let b = a.select(&[0, 2], &[1]);

        assert_eq!(b.into_vec(), vec![1, 5]);
    }

    #[test]
    fn matrix_diag() {
        let a = Matrix::new(3, 3, vec![1., 3., 5., 2., 4., 7., 1., 1., 0.]);

        let b = a.is_diag();

        assert!(!b);

        let c = Matrix::new(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 3.]);
        let d = c.is_diag();

        assert!(d);
    }

    #[test]
    fn matrix_det() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);
        let b = a.det();

        assert_eq!(b, 1.);

        let c = Matrix::new(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let d = c.det();

        assert_eq!(d, 0.);

        let e = Matrix::<f64>::new(5,
                                   5,
                                   vec![1., 2., 3., 4., 5., 3., 0., 4., 5., 6., 2., 1., 2., 3.,
                                        4., 0., 0., 0., 6., 5., 0., 0., 0., 5., 6.]);

        let f = e.det();

        println!("det is {0}", f);
        let error = abs(f - 99.);
        assert!(error < 1e-10);
    }

    #[test]
    fn matrix_solve() {
        let a = Matrix::new(2, 2, vec![2., 3., 1., 2.]);

        let y = Vector::new(vec![8., 5.]);

        let x = a.solve(y).unwrap();

        assert_eq!(x.size(), 2);

        assert_eq!(x[0], 1.);
        assert_eq!(x[1], 2.);
    }

    #[test]
    fn create_mat_zeros() {
        let a = Matrix::<f32>::zeros(10, 10);

        assert_eq!(a.rows(), 10);
        assert_eq!(a.cols(), 10);

        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(a[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn create_mat_identity() {
        let a = Matrix::<f32>::identity(4);

        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 4);

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 2]], 1.0);
        assert_eq!(a[[3, 3]], 1.0);

        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[2, 1]], 0.0);
        assert_eq!(a[[3, 0]], 0.0);
    }

    #[test]
    fn create_mat_diag() {
        let a = Matrix::from_diag(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 4);

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 2.0);
        assert_eq!(a[[2, 2]], 3.0);
        assert_eq!(a[[3, 3]], 4.0);

        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[2, 1]], 0.0);
        assert_eq!(a[[3, 0]], 0.0);
    }

    #[test]
    fn transpose_mat() {
        let a = Matrix::new(5, 2, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);

        let c = a.transpose();

        assert_eq!(c.cols(), a.rows());
        assert_eq!(c.rows(), a.cols());

        assert_eq!(a[[0, 0]], c[[0, 0]]);
        assert_eq!(a[[1, 0]], c[[0, 1]]);
        assert_eq!(a[[2, 0]], c[[0, 2]]);
        assert_eq!(a[[3, 0]], c[[0, 3]]);
        assert_eq!(a[[4, 0]], c[[0, 4]]);
        assert_eq!(a[[0, 1]], c[[1, 0]]);
        assert_eq!(a[[1, 1]], c[[1, 1]]);
        assert_eq!(a[[2, 1]], c[[1, 2]]);
        assert_eq!(a[[3, 1]], c[[1, 3]]);
        assert_eq!(a[[4, 1]], c[[1, 4]]);

    }
}
