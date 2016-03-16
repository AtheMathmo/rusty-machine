//! Slices for the `Matrix` struct.
//!
//! These slices provide a view into the matrix data.
//! The view must be a contiguous block of the matrix.
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::linalg::matrix::slice::MatrixSlice;
//!
//! let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
//! 
//! // Manually create our slice - [[4,5],[7,8]].
//! let mat_slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
//!
//! // We can perform arithmetic with slices.
//! let new_mat = &mat_slice * &mat_slice;
//! ```

use super::Matrix;

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use std::marker::PhantomData;
use std::mem;
use libnum::{One, Zero};

use linalg::utils;

/// Trait for Matrix Slices.
pub trait BaseSlice<T> {
    /// Rows in the slice.
    fn rows(&self) -> usize;

    /// Columns in the slice.
    fn cols(&self) -> usize;

    /// Top left index of the slice.
    fn as_ptr(&self) -> *const T;

    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T;
}

/// A MatrixSlice
///
/// This struct provides a slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixSlice<T> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    row_stride: usize,
}

/// A mutable MatrixSlice
///
/// This struct provides a mutable slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug)]
pub struct MatrixSliceMut<T> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
}

impl<T> BaseSlice<T> for MatrixSlice<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        &*(self.ptr.offset((index[0] * self.row_stride + index[1]) as isize))
    }
}

impl<T> BaseSlice<T> for MatrixSliceMut<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        &*(self.ptr.offset((index[0] * self.row_stride + index[1]) as isize))
    }
}

impl<T> MatrixSlice<T> {
    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixSlice<T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");
        unsafe {
            MatrixSlice {
                ptr: mat.data().get_unchecked(start[0] * mat.cols + start[1]) as *const T,
                rows: rows,
                cols: cols,
                row_stride: mat.cols,
            }
        }
    }

    /// Returns an iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&self) -> SliceIter<T> {
        SliceIter {
            slice_start: self.ptr,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_diff: self.row_stride as isize - self.cols as isize + 1,
            _marker: PhantomData::<&T>,
        }
    }
}

impl<T: Copy> MatrixSlice<T> {
    /// Convert the matrix slice into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        let slice_data = self.iter().map(|v| *v).collect::<Vec<T>>();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: slice_data,
        }
    }
}

impl<T> MatrixSliceMut<T> {

    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &mut Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixSliceMut<T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");

        let mat_cols = mat.cols();

        unsafe {
            MatrixSliceMut {
                ptr: mat.mut_data().get_unchecked_mut(start[0] * mat_cols + start[1]) as *mut T,
                rows: rows,
                cols: cols,
                row_stride: mat_cols,
            }
        }
    }

    /// Returns an iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&self) -> SliceIter<T> {
        let row_diff = self.row_stride as isize - self.cols as isize + 1;
        SliceIter {
            slice_start: self.ptr as *const T,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_diff: row_diff,
            _marker: PhantomData::<&T>,
        }
    }

    /// Returns a mutable iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    ///
    /// {
    ///     let mut slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    ///
    ///     for d in slice.iter_mut() {
    ///         *d = *d + 2;
    ///     }
    /// }
    ///
    /// // Only the matrix slice is updated.
    /// assert_eq!(a.into_vec(), vec![0,1,2,3,6,7,6,9,10]);
    /// ```
    pub fn iter_mut(&mut self) -> SliceIterMut<T> {
        let row_diff = self.row_stride as isize - self.cols as isize + 1;
        SliceIterMut {
            slice_start: self.ptr,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_diff: row_diff,
            _marker: PhantomData::<&mut T>,
        }
    }
}


impl<T: Copy> MatrixSliceMut<T> {
    /// Convert the matrix slice into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        let slice_data = self.iter().map(|v| *v).collect::<Vec<T>>();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: slice_data,
        }
    }
}

/// Iterator for the MatrixSlice
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIter<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_diff: isize,
    _marker: PhantomData<&'a T>,
}

/// Iterates over the matrix slice data in row-major order.
impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // Set the position of the next element
        if self.row_pos < self.slice_rows {
            unsafe {
                let ret_ptr = self.slice_start;
                // If end of row, set to start of next row
                if self.col_pos == self.slice_cols - 1 {
                    self.row_pos += 1usize;
                    self.col_pos = 0usize;
                    self.slice_start = self.slice_start.offset(self.row_diff);
                } else {
                    self.col_pos += 1usize;
                    self.slice_start = self.slice_start.offset(1);
                }

                Some(mem::transmute(ret_ptr))
            }
        } else {
            None
        }
    }
}

/// Iterator for MatrixSliceMut.
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIterMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    col_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_diff: isize,
    _marker: PhantomData<&'a mut T>,
}


/// Iterates over the matrix slice data in row-major order.
impl<'a, T> Iterator for SliceIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        // Set the position of the next element
        if self.row_pos < self.slice_rows {
            unsafe {
                let ret_ptr = self.slice_start;
                // If end of row, set to start of next row
                if self.col_pos == self.slice_cols - 1 {
                    self.row_pos += 1usize;
                    self.col_pos = 0usize;
                    self.slice_start = self.slice_start.offset(self.row_diff);
                } else {
                    self.col_pos += 1usize;
                    self.slice_start = self.slice_start.offset(1);
                }
                Some(mem::transmute(ret_ptr))
            }
        } else {
            None
        }
    }
}

/// Indexes matrix slice.
///
/// Takes row index first then column.
impl<'a, T> Index<[usize; 2]> for MatrixSlice<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &*(self.as_ptr().offset((idx[0] * self.row_stride + idx[1]) as isize))
        }
    }
}

/// Multiplies matrix slice by scalar.
impl<'a, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

/// Multiplies matrix slice by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'b T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

/// Multiplies matrix slice by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

/// Multiplies matrix slice by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'c T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.iter().map(|v| (*v) * (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Multiplies matrix slice by matrix.
impl<'a, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix slice by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix slice by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix slice by matrix.
impl<'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c Matrix<T>> for &'b MatrixSlice<T> {
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
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) + *self.get_unchecked([i,k]) * *m.data().get_unchecked(k*m.cols + j);
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

/// Multiplies matrix slice by matrix slice.
impl<'a, 'b, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixSlice<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix slice by matrix slice.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixSlice<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix slice by matrix slice.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixSlice<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix slice by matrix slice.
impl<'a, 'b, 'c, 'd, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'d MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        unsafe {
            for i in 0..self.rows
            {
                for k in 0..m.rows
                {
                    for j in 0..m.cols
                    {
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) +
                        							*self.get_unchecked([i,k]) *
                        							*m.get_unchecked([k,j]);
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

/// Adds scalar to matrix slice.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds scalar to matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds scalar to matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds scalar to matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        let new_data = self.iter().map(|v| (*v) + (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix slice.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c Matrix<T>> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = *x + y });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix slice to matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixSlice<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix slice to matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixSlice<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix slice to matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, f: &MatrixSlice<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix slice to matrix slice.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Add<T, Output = T>> Add<&'d MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn add(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut res_data : Vec<T> = self.iter().map(|x| *x).collect();
        let m_data : Vec<T> = m.iter().map(|x| *x).collect();

        utils::in_place_vec_bin_op(&mut res_data, &m_data, |x, &y| { *x = *x + y });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: res_data,
        }
    }
}

/// Subtracts scalar from matrix slice.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts scalar from matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'a T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts scalar from matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts scalar from matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        let new_data = self.iter().map(|v| (*v) - *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts matrix from matrix slice.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix from matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix from matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'b Matrix<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix from matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c Matrix<T>> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = *x - y });
        

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts matrix slice from matrix slice.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixSlice<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix slice from matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixSlice<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix slice from matrix slice.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c MatrixSlice<T>> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, f: &MatrixSlice<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix slice from matrix slice.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'d MatrixSlice<T>> for &'c MatrixSlice<T> {
    type Output = Matrix<T>;

    fn sub(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut res_data : Vec<T> = self.iter().map(|x| *x).collect();
        let m_data : Vec<T> = m.iter().map(|x| *x).collect();

        utils::in_place_vec_bin_op(&mut res_data, &m_data, |x, &y| { *x = *x - y });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: res_data,
        }
    }
}

/// Divides matrix slice by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        (&self) / (&f)
    }
}

/// Divides matrix slice by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

/// Divides matrix slice by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'b T> for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

/// Divides matrix slice by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'c T> for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        assert!(*f != T::zero());

        let new_data = self.iter().map(|v| (*v) / *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix slice.
impl<'a, T: Neg<Output = T> + Copy> Neg for MatrixSlice<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix slice.
impl<'a, 'b, T: Neg<Output = T> + Copy> Neg for &'b MatrixSlice<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

#[cfg(test)]
mod tests {
	use super::MatrixSlice;
	use super::super::Matrix;

	#[test]
	#[should_panic]
	fn make_slice_bad_dim() {
		let a = Matrix::new(3,3, vec![2.0; 9]);
		let _ = MatrixSlice::from_matrix(&a, [1,1], 3, 2);
	}

	#[test]
	fn add_slice() {
		let a = 3.0;
		let b = Matrix::new(3,3, vec![2.0; 9]);
		let c = Matrix::new(2,2, vec![1.0; 4]);

		let d = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m_1 = &d + a;
		assert_eq!(m_1.into_vec(), vec![5.0; 4]);

		let m_2 = &d + c;
		assert_eq!(m_2.into_vec(), vec![3.0; 4]);

		let m_3 = &d + &d;
		assert_eq!(m_3.into_vec(), vec![4.0; 4]);		
	}

	#[test]
	fn sub_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d = MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d - a;
		assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

		let m_2 = &d - b;
		assert_eq!(m_2.into_vec(), vec![1.0; 4]);

		let m_3 = &d - &d;
		assert_eq!(m_3.into_vec(), vec![0.0; 4]);
	}

	#[test]
	fn mul_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d= MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d * a;
		assert_eq!(m_1.into_vec(), vec![6.0; 4]);

		let m_2 = &d * b;
		assert_eq!(m_2.into_vec(), vec![4.0; 4]);

		let m_3 = &d * d;
		assert_eq!(m_3.into_vec(), vec![8.0; 4]);
	}

	#[test]
	fn div_slice() {
		let a = 3.0;

		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = c / a;
		assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
	}

	#[test]
	fn neg_slice() {
		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = -c;
		assert_eq!(m.into_vec(), vec![-2.0;4]);
	}

	#[test]
	fn index_slice() {
		let b = Matrix::new(3,3, (0..9).collect());

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);
		
		assert_eq!(c[[0,0]], 4);
		assert_eq!(c[[0,1]], 5);
		assert_eq!(c[[1,0]], 7);
		assert_eq!(c[[1,1]], 8);
	}
}