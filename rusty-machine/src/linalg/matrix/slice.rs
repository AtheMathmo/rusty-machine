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

use std::ops::{Mul, Add, Div, Sub, Index, IndexMut, Neg};
use std::marker::PhantomData;
use std::mem;
use libnum::Zero;

use linalg::utils;

/// Trait for Matrix Slices.
pub trait BaseSlice<T> {
    /// Rows in the slice.
    fn rows(&self) -> usize;

    /// Columns in the slice.
    fn cols(&self) -> usize;

    /// Top left index of the slice.
    fn as_ptr(&self) -> *const T;

    /// Get a reference to a point in the slice without bounds checking.
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

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self,
                            start: [usize; 2],
                            rows: usize,
                            cols: usize)
                            -> MatrixSlice<T> {
        assert!(start[0] + rows <= self.rows,
            "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= self.cols,
                "View dimensions exceed matrix dimensions.");

        unsafe {
            self.ptr = self.ptr.offset((start[0] * self.cols + start[1]) as isize);
        }
        self.rows = rows;
        self.cols = cols;

        self
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

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self,
                            start: [usize; 2],
                            rows: usize,
                            cols: usize)
                            -> MatrixSliceMut<T> {
        assert!(start[0] + rows <= self.rows,
            "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= self.cols,
                "View dimensions exceed matrix dimensions.");

        unsafe {
            self.ptr = self.ptr.offset((start[0] * self.cols + start[1]) as isize);
        }
        self.rows = rows;
        self.cols = cols;

        self
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
impl<T> Index<[usize; 2]> for MatrixSlice<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &*(self.ptr.offset((idx[0] * self.row_stride + idx[1]) as isize))
        }
    }
}

/// Indexes mutable matrix slice.
///
/// Takes row index first then column.
impl<T> Index<[usize; 2]> for MatrixSliceMut<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &*(self.ptr.offset((idx[0] * self.row_stride + idx[1]) as isize))
        }
    }
}

/// Indexes mutable matrix slice.
///
/// Takes row index first then column.
impl<T> IndexMut<[usize; 2]> for MatrixSliceMut<T> {

    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &mut *(self.ptr.offset((idx[0] * self.row_stride + idx[1]) as isize))
        }
    }
}

macro_rules! impl_bin_op_scalar_slice (
    ($trt:ident, $op:ident, $slice:ident) => (

impl<T: Copy + $trt<T, Output=T>> $trt<T> for $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<&'a T> for $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        (&self).$op(f)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<T> for &'a $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

impl<'a, 'b, T: Copy + $trt<T, Output=T>> $trt<&'b T> for &'a $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.iter().map(|v| (*v).$op(*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_scalar_slice!(Mul, mul, MatrixSlice);
impl_bin_op_scalar_slice!(Mul, mul, MatrixSliceMut);
impl_bin_op_scalar_slice!(Div, div, MatrixSlice);
impl_bin_op_scalar_slice!(Div, div, MatrixSliceMut);
impl_bin_op_scalar_slice!(Add, add, MatrixSlice);
impl_bin_op_scalar_slice!(Add, add, MatrixSliceMut);
impl_bin_op_scalar_slice!(Sub, sub, MatrixSlice);
impl_bin_op_scalar_slice!(Sub, sub, MatrixSliceMut);


macro_rules! impl_mat_mul (
    ($mat_1:ident, $mat_2:ident) => (

impl<T: Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$mat_2<T>> for $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $mat_2<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

impl<'a, T: Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'a $mat_2<T>> for $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$mat_2<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

impl<'a, T: Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$mat_2<T>> for &'a $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $mat_2<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

impl<'a, 'b, T: Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'b $mat_2<T>> for &'a $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$mat_2<T>) -> Matrix<T> {
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        unsafe {
            for i in 0..self.rows
            {
                for k in 0..m.rows
                {
                    for j in 0..m.cols
                    {
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) + *self.get_unchecked([i,k]) * *m.get_unchecked([k,j]);
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
    );
);

impl_mat_mul!(MatrixSlice, Matrix);
impl_mat_mul!(MatrixSlice, MatrixSlice);
impl_mat_mul!(MatrixSlice, MatrixSliceMut);
impl_mat_mul!(MatrixSliceMut, Matrix);
impl_mat_mul!(MatrixSliceMut, MatrixSlice);
impl_mat_mul!(MatrixSliceMut, MatrixSliceMut);
impl_mat_mul!(Matrix, MatrixSlice);
impl_mat_mul!(Matrix, MatrixSliceMut);

macro_rules! impl_bin_op_slice (
    ($trt:ident, $op:ident, $slice_1:ident, $slice_2:ident) => (

impl<T: Copy + $trt<T, Output=T>> $trt<$slice_2<T>> for $slice_1<T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (&self).$op(&s)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<$slice_2<T>> for &'a $slice_1<T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (self).$op(&s)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<&'a $slice_2<T>> for $slice_1<T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

impl<'a, 'b, T: Copy + $trt<T, Output=T>> $trt<&'b $slice_2<T>> for &'a $slice_1<T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut res_data : Vec<T> = self.iter().map(|x| *x).collect();
        let s_data : Vec<T> = s.iter().map(|x| *x).collect();

        utils::in_place_vec_bin_op(&mut res_data, &s_data, |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: res_data,
        }
    }
}
    );
);

impl_bin_op_slice!(Add, add, MatrixSlice, MatrixSlice);
impl_bin_op_slice!(Add, add, MatrixSliceMut, MatrixSlice);
impl_bin_op_slice!(Add, add, MatrixSlice, MatrixSliceMut);
impl_bin_op_slice!(Add, add, MatrixSliceMut, MatrixSliceMut);

impl_bin_op_slice!(Sub, sub, MatrixSlice, MatrixSlice);
impl_bin_op_slice!(Sub, sub, MatrixSliceMut, MatrixSlice);
impl_bin_op_slice!(Sub, sub, MatrixSlice, MatrixSliceMut);
impl_bin_op_slice!(Sub, sub, MatrixSliceMut, MatrixSliceMut);

macro_rules! impl_bin_op_mat_slice (
    ($trt:ident, $op:ident, $slice:ident) => (

impl<T: Copy + $trt<T, Output=T>> $trt<Matrix<T>> for $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        (&self).$op(m)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<Matrix<T>> for &'a $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        self.$op(&m)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<&'a Matrix<T>> for $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        (&self).$op(m)
    }
}

impl<'a, 'b, T: Copy + $trt<T, Output=T>> $trt<&'b Matrix<T>> for &'a $slice<T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

impl<T: Copy + $trt<T, Output=T>> $trt<$slice<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<$slice<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        self.$op(&s)
    }
}

impl<'a, T: Copy + $trt<T, Output=T>> $trt<&'a $slice<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

impl<'a, 'b, T: Copy + $trt<T, Output=T>> $trt<&'b $slice<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = s.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, self.data(), |x, &y| { *x = (y).$op(*x) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_mat_slice!(Add, add, MatrixSlice);
impl_bin_op_mat_slice!(Add, add, MatrixSliceMut);

impl_bin_op_mat_slice!(Sub, sub, MatrixSlice);
impl_bin_op_mat_slice!(Sub, sub, MatrixSliceMut);


macro_rules! impl_neg_slice (
    ($slice:ident) => (

/// Gets negative of matrix slice.
impl<T: Neg<Output = T> + Copy> Neg for $slice<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        - &self
    }
}

/// Gets negative of matrix slice.
impl<'a, T: Neg<Output = T> + Copy> Neg for &'a $slice<T> {
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

    );
);

impl_neg_slice!(MatrixSlice);
impl_neg_slice!(MatrixSliceMut);

#[cfg(test)]
mod tests {
	use super::MatrixSlice;
    use super::MatrixSliceMut;
    use super::BaseSlice;
	use super::super::Matrix;

	#[test]
	#[should_panic]
	fn make_slice_bad_dim() {
		let a = Matrix::new(3,3, vec![2.0; 9]);
		let _ = MatrixSlice::from_matrix(&a, [1,1], 3, 2);
	}

    #[test]
    fn make_slice() {
        let a = Matrix::new(3,3, vec![2.0; 9]);
        let b = MatrixSlice::from_matrix(&a, [1,1], 2, 2);

        assert_eq!(b.rows(), 2);
        assert_eq!(b.cols(), 2);
    }

    #[test]
    fn reslice() {
        let mut a = Matrix::new(4,4, (0..16).collect());
        let b = MatrixSlice::from_matrix(&a, [1,1], 3, 3);
        {
            let c = b.reslice([0,1], 2, 2);

            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);

            assert_eq!(c[[0,0]], 6);
            assert_eq!(c[[0,1]], 7);
            assert_eq!(c[[1,0]], 10);
            assert_eq!(c[[1,1]], 11);
        }

        let b = MatrixSliceMut::from_matrix(&mut a, [1,1], 3, 3);

        let c = b.reslice([0,1], 2, 2);

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        assert_eq!(c[[0,0]], 6);
        assert_eq!(c[[0,1]], 7);
        assert_eq!(c[[1,0]], 10);
        assert_eq!(c[[1,1]], 11);
    }

	#[test]
	fn add_slice() {
		let a = 3.0;
		let mut b = Matrix::new(3,3, vec![2.0; 9]);
		let c = Matrix::new(2,2, vec![1.0; 4]);

		let d = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m_1 = &d + a.clone();
		assert_eq!(m_1.into_vec(), vec![5.0; 4]);

        let m_2 = c.clone() + &d;
        assert_eq!(m_2.into_vec(), vec![3.0; 4]);

		let m_3 = &d + c.clone();
		assert_eq!(m_3.into_vec(), vec![3.0; 4]);

		let m_4 = &d + &d;
		assert_eq!(m_4.into_vec(), vec![4.0; 4]);		

        let e = MatrixSliceMut::from_matrix(&mut b, [1,1], 2, 2);

        let m_1 = &e + a.clone();
        assert_eq!(m_1.into_vec(), vec![5.0; 4]);

        let m_2 = c.clone() + &e;
        assert_eq!(m_2.into_vec(), vec![3.0; 4]);

        let m_3 = &e + c;
        assert_eq!(m_3.into_vec(), vec![3.0; 4]);

        let m_4 = &e + &e;
        assert_eq!(m_4.into_vec(), vec![4.0; 4]);
	}

	#[test]
	fn sub_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let mut c = Matrix::new(3,3, vec![2.0; 9]);

		let d = MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d - a.clone();
		assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

        let m_2 = b.clone() - &d;
        assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

		let m_3 = &d - b.clone();
		assert_eq!(m_3.into_vec(), vec![1.0; 4]);

		let m_4 = &d - &d;
		assert_eq!(m_4.into_vec(), vec![0.0; 4]);

        let e = MatrixSliceMut::from_matrix(&mut c, [1,1], 2, 2);

        let m_1 = &e - a;
        assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

        let m_2 = b.clone() - &e;
        assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

        let m_3 = &e - b;
        assert_eq!(m_3.into_vec(), vec![1.0; 4]);

        let m_4 = &e - &e;
        assert_eq!(m_4.into_vec(), vec![0.0; 4]);
	}

	#[test]
	fn mul_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let mut c = Matrix::new(3,3, vec![2.0; 9]);

		let d = MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d * a.clone();
		assert_eq!(m_1.into_vec(), vec![6.0; 4]);

		let m_2 = &d * b.clone();
		assert_eq!(m_2.into_vec(), vec![4.0; 4]);

		let m_3 = &d * &d;
		assert_eq!(m_3.into_vec(), vec![8.0; 4]);

        let e = MatrixSliceMut::from_matrix(&mut c, [1,1], 2, 2);

        let m_1 = &e * a;
        assert_eq!(m_1.into_vec(), vec![6.0; 4]);

        let m_2 = &e * b;
        assert_eq!(m_2.into_vec(), vec![4.0; 4]);

        let m_3 = &e * &e;
        assert_eq!(m_3.into_vec(), vec![8.0; 4]);
	}

	#[test]
	fn div_slice() {
		let a = 3.0;

		let mut b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = c / a;
		assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);

        let d = MatrixSliceMut::from_matrix(&mut b, [1,1], 2, 2);

        let m = d / a;
        assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
	}

	#[test]
	fn neg_slice() {
		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = -c;
		assert_eq!(m.into_vec(), vec![-2.0;4]);

        let mut b = Matrix::new(3,3, vec![2.0; 9]);

        let c = MatrixSliceMut::from_matrix(&mut b, [1,1], 2, 2);

        let m = -c;
        assert_eq!(m.into_vec(), vec![-2.0;4]);
	}

	#[test]
	fn index_slice() {
		let mut b = Matrix::new(3,3, (0..9).collect());

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);
		
		assert_eq!(c[[0,0]], 4);
		assert_eq!(c[[0,1]], 5);
		assert_eq!(c[[1,0]], 7);
		assert_eq!(c[[1,1]], 8);

        let mut c = MatrixSliceMut::from_matrix(&mut b, [1,1], 2, 2);
        
        assert_eq!(c[[0,0]], 4);
        assert_eq!(c[[0,1]], 5);
        assert_eq!(c[[1,0]], 7);
        assert_eq!(c[[1,1]], 8);

        c[[0,0]] = 9;

        assert_eq!(c[[0,0]], 9);
        assert_eq!(c[[0,1]], 5);
        assert_eq!(c[[1,0]], 7);
        assert_eq!(c[[1,1]], 8);
	}

    #[test]
    fn slice_into_matrix() {
        let mut a = Matrix::new(3,3, vec![2.0; 9]);

        let b = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
        let c = b.into_matrix();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        let d = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
        let e = d.into_matrix();
        assert_eq!(e.rows(), 2);
        assert_eq!(e.cols(), 2);
    }
}