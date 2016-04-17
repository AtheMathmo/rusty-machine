//! Slices for the `Matrix` struct.
//!
//! These slices provide a view into the matrix data.
//! The view must be a contiguous block of the matrix.
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::linalg::matrix::MatrixSlice;
//!
//! let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
//! 
//! // Manually create our slice - [[4,5],[7,8]].
//! let mat_slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
//!
//! // We can perform arithmetic with slices.
//! let new_mat = &mat_slice * &mat_slice;
//! ```

use super::{Matrix, MatrixSlice, MatrixSliceMut, Axes};

use std::marker::PhantomData;
use std::mem;

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
    /// use rusty_machine::linalg::matrix::MatrixSlice;
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
    /// use rusty_machine::linalg::matrix::MatrixSlice;
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
    /// use rusty_machine::linalg::matrix::MatrixSlice;
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

    pub fn split_at(self, mid: usize, axis: Axes) -> (MatrixSlice<T>, MatrixSlice<T>) {
        let slice_1 : MatrixSlice<T>;
        let slice_2 : MatrixSlice<T>;

        let self_cols = self.cols;
        let self_rows = self.rows;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                slice_1 = self.reslice([0,0], mid, self_cols);
                slice_2 = self.reslice([mid,0], self_rows - mid, self_cols);
            },
            Axes::Col => {
                assert!(mid < self.cols);

                slice_1 = self.reslice([0,0], self_rows, mid);
                slice_2 = self.reslice([0,mid], self_rows, self_cols - mid);
            }
        }

        (slice_1, slice_2)
    }
}

impl<T> MatrixSliceMut<T> {

    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::MatrixSliceMut;
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
    /// use rusty_machine::linalg::matrix::MatrixSliceMut;
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
    /// use rusty_machine::linalg::matrix::MatrixSliceMut;
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
    /// use rusty_machine::linalg::matrix::MatrixSliceMut;
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

    pub fn split_at(self, mid: usize, axis: Axes) -> (MatrixSliceMut<T>, MatrixSliceMut<T>) {
        let slice_1 : MatrixSliceMut<T>;
        let slice_2 : MatrixSliceMut<T>;

        let self_cols = self.cols;
        let self_rows = self.rows;

        match axis {
            Axes::Row => {
                assert!(mid < self.rows);

                slice_1 = self.clone().reslice([0,0], mid, self_cols);
                slice_2 = self.reslice([mid,0], self_rows - mid, self_cols);
            },
            Axes::Col => {
                assert!(mid < self.cols);

                slice_1 = self.clone().reslice([0,0], self_rows, mid);
                slice_2 = self.reslice([0,mid], self_rows, self_cols - mid);
            }
        }

        (slice_1, slice_2)
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

#[cfg(test)]
mod tests {
    use super::BaseSlice;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;
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