//! Slices for the `Matrix` struct.
//!
//! These slices provide a view into the matrix data.
//! The view must be a contiguous block of the matrix.
//!
//! ```
//! use rulinalg::matrix::Matrix;
//! use rulinalg::matrix::MatrixSlice;
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
use super::MatrixSlice;
use super::MatrixSliceMut;
use utils;

use std::marker::PhantomData;
use std::mem;

/// Trait for Matrix Slices.
pub trait BaseSlice<T> {
    /// Rows in the slice.
    fn rows(&self) -> usize;

    /// Columns in the slice.
    fn cols(&self) -> usize;

    /// Row stride in the slice.
    fn row_stride(&self) -> usize;

    /// Top left index of the slice.
    fn as_ptr(&self) -> *const T;

    /// Get a reference to a point in the slice without bounds checking.
    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T;
}

impl<'a, T> BaseSlice<T> for MatrixSlice<'a, T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn row_stride(&self) -> usize {
        self.row_stride
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        &*(self.ptr.offset((index[0] * self.row_stride + index[1]) as isize))
    }
}

impl<'a, T> BaseSlice<T> for MatrixSliceMut<'a, T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn row_stride(&self) -> usize {
        self.row_stride
    }

    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    unsafe fn get_unchecked(&self, index: [usize; 2]) -> &T {
        &*(self.ptr.offset((index[0] * self.row_stride + index[1]) as isize))
    }
}

impl<'a, T> MatrixSlice<'a, T> {
    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a Matrix<T>,
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
                marker: PhantomData::<&'a T>,
            }
        }
    }

    /// Creates a matrix slice from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let mut a = vec![4.0; 16];
    ///
    /// unsafe {
    ///     // Create a matrix slice with 3 rows, and 3 cols
    ///     // The row stride of 4 specifies the distance between the start of each row in the data.
    ///     let b = MatrixSlice::from_raw_parts(a.as_ptr(), 3, 3, 4);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `row_stride * rows`.
    /// If not then other operations will produce undefined behaviour.
    ///
    /// Additionally `cols` should be less than the `row_stride`. It is possible to use this
    /// function safely whilst violating this condition. So long as
    /// `max(cols, row_stride) * rows` is less than the data size.
    pub unsafe fn from_raw_parts(ptr: *const T,
                                 rows: usize,
                                 cols: usize,
                                 row_stride: usize)
                                 -> MatrixSlice<'a, T> {
        MatrixSlice {
            ptr: ptr,
            rows: rows,
            cols: cols,
            row_stride: row_stride,
            marker: PhantomData::<&'a T>,
        }
    }

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self, start: [usize; 2], rows: usize, cols: usize) -> MatrixSlice<'a, T> {
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
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter {
            slice_start: self.ptr,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride,
            _marker: PhantomData::<&'a T>,
        }
    }
}

impl<'a, T: Copy> MatrixSlice<'a, T> {
    /// Convert the matrix slice into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        self.iter_rows().collect::<Matrix<T>>()
    }
}

impl<'a, T> MatrixSliceMut<'a, T> {
    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a mut Matrix<T>,
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
                marker: PhantomData::<&'a mut T>,
            }
        }
    }

    /// Creates a mutable matrix slice from raw parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = vec![4.0; 16];
    ///
    /// unsafe {
    ///     // Create a mutable matrix slice with 3 rows, and 3 cols
    ///     // The row stride of 4 specifies the distance between the start of each row in the data.
    ///     let b = MatrixSliceMut::from_raw_parts(a.as_mut_ptr(), 3, 3, 4);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The pointer must be followed by a contiguous slice of data larger than `row_stride * rows`.
    /// If not then other operations will produce undefined behaviour.
    ///
    /// Additionally `cols` should be less than the `row_stride`. It is possible to use this
    /// function safely whilst violating this condition. So long as
    /// `max(cols, row_stride) * rows` is less than the data size.
    pub unsafe fn from_raw_parts(ptr: *mut T,
                                 rows: usize,
                                 cols: usize,
                                 row_stride: usize)
                                 -> MatrixSliceMut<'a, T> {
        MatrixSliceMut {
            ptr: ptr,
            rows: rows,
            cols: cols,
            row_stride: row_stride,
            marker: PhantomData::<&'a mut T>,
        }
    }

    /// Produce a matrix slice from an existing matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    /// let new_slice = slice.reslice([0,0], 1, 1);
    /// ```
    pub fn reslice(mut self, start: [usize; 2], rows: usize, cols: usize) -> MatrixSliceMut<'a, T> {
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
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
    ///
    /// let mut a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSliceMut::from_matrix(&mut a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter {
            slice_start: self.ptr as *const T,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride,
            _marker: PhantomData::<&T>,
        }
    }

    /// Returns a mutable iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    /// use rulinalg::matrix::MatrixSliceMut;
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
    pub fn iter_mut(&mut self) -> SliceIterMut<'a, T> {
        SliceIterMut {
            slice_start: self.ptr,
            row_pos: 0,
            col_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride,
            _marker: PhantomData::<&mut T>,
        }
    }
}

impl<'a, T: Copy> MatrixSliceMut<'a, T> {
    /// Convert the matrix slice into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        self.iter_rows().collect::<Matrix<T>>()
    }

    /// Sets the underlying matrix data to the target data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    ///
    /// let mut mat = Matrix::<f32>::zeros(4,4);
    /// let one_block = Matrix::<f32>::ones(2,2);
    ///
    /// // Get a mutable slice of the upper left 2x2 block.
    /// let mat_block = MatrixSliceMut::from_matrix(&mut mat, [0,0], 2, 2);
    ///
    /// // Set the upper left 2x2 block to be ones.
    /// mat_block.set_to(one_block.as_slice());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `target` are not the same.
    pub fn set_to(self, target: MatrixSlice<T>) {
        // TODO: Should this method take an Into<MatrixSlice> or something similar?
        // So we can use `Matrix` and `MatrixSlice` and `MatrixSliceMut`.
        assert!(self.rows == target.rows,
                "Target has different row count to self.");
        assert!(self.cols == target.cols,
                "Target has different column count to self.");
        for (s, t) in self.iter_rows_mut().zip(target.iter_rows()) {
            // Vectorized assignment per row.
            utils::in_place_vec_bin_op(s, t, |x, &y| *x = y);
        }
    }
}

/// Iterator for `MatrixSlice`
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
    row_stride: usize,
    _marker: PhantomData<&'a T>,
}

/// Iterator for `MatrixSliceMut`.
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
    row_stride: usize,
    _marker: PhantomData<&'a mut T>,
}

macro_rules! impl_slice_iter (
    ($slice_iter:ident, $data_type:ty) => (
/// Iterates over the matrix slice data in row-major order.
impl<'a, T> Iterator for $slice_iter<'a, T> {
    type Item = $data_type;

    fn next(&mut self) -> Option<Self::Item> {
// Set the position of the next element
        if self.row_pos < self.slice_rows {
            unsafe {
                let iter_ptr = self.slice_start.offset((
                                self.row_pos * self.row_stride + self.col_pos)
                                as isize);

// If end of row, set to start of next row
                if self.col_pos == self.slice_cols - 1 {
                    self.row_pos += 1usize;
                    self.col_pos = 0usize;
                } else {
                    self.col_pos += 1usize;
                }

                Some(mem::transmute(iter_ptr))
            }
        } else {
            None
        }
    }
}
    );
);

impl_slice_iter!(SliceIter, &'a T);
impl_slice_iter!(SliceIterMut, &'a mut T);

#[cfg(test)]
mod tests {
    use super::BaseSlice;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;
    use super::super::Matrix;

    #[test]
    #[should_panic]
    fn make_slice_bad_dim() {
        let a = Matrix::new(3, 3, vec![2.0; 9]);
        let _ = MatrixSlice::from_matrix(&a, [1, 1], 3, 2);
    }

    #[test]
    fn make_slice() {
        let a = Matrix::new(3, 3, vec![2.0; 9]);
        let b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);

        assert_eq!(b.rows(), 2);
        assert_eq!(b.cols(), 2);
    }

    #[test]
    fn reslice() {
        let mut a = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());

        {
            let b = MatrixSlice::from_matrix(&a, [1, 1], 3, 3);
            let c = b.reslice([0, 1], 2, 2);

            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);

            assert_eq!(c[[0, 0]], 6);
            assert_eq!(c[[0, 1]], 7);
            assert_eq!(c[[1, 0]], 10);
            assert_eq!(c[[1, 1]], 11);
        }

        let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 3, 3);

        let c = b.reslice([0, 1], 2, 2);

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        assert_eq!(c[[0, 0]], 6);
        assert_eq!(c[[0, 1]], 7);
        assert_eq!(c[[1, 0]], 10);
        assert_eq!(c[[1, 1]], 11);
    }

    #[test]
    fn slice_into_matrix() {
        let mut a = Matrix::new(3, 3, vec![2.0; 9]);

        {
            let b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);
            let c = b.into_matrix();
            assert_eq!(c.rows(), 2);
            assert_eq!(c.cols(), 2);
        }

        let d = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
        let e = d.into_matrix();
        assert_eq!(e.rows(), 2);
        assert_eq!(e.cols(), 2);
    }
}
