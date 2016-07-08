use std::iter::{ExactSizeIterator, FromIterator};
use std::marker::PhantomData;
use std::slice;

use super::{Matrix, MatrixSlice, MatrixSliceMut};
use super::slice::{SliceIter, SliceIterMut};

/// Row iterator.
#[derive(Debug)]
pub struct Rows<'a, T: 'a> {
    slice_start: *const T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a T>,
}

/// Mutable row iterator.
#[derive(Debug)]
pub struct RowsMut<'a, T: 'a> {
    slice_start: *mut T,
    row_pos: usize,
    slice_rows: usize,
    slice_cols: usize,
    row_stride: isize,
    _marker: PhantomData<&'a mut T>,
}

macro_rules! impl_iter_rows (
    ($rows:ident, $row_type:ty, $slice_from_parts:ident) => (

/// Iterates over the rows in the matrix.
impl<'a, T> Iterator for $rows<'a, T> {
    type Item = $row_type;

    fn next(&mut self) -> Option<Self::Item> {
// Check if we have reached the end
        if self.row_pos < self.slice_rows {
            let row: $row_type;
            unsafe {
// Get pointer and create a slice from raw parts
                let ptr = self.slice_start.offset(self.row_pos as isize * self.row_stride);
                row = slice::$slice_from_parts(ptr, self.slice_cols);
            }

            self.row_pos += 1;
            Some(row)
        } else {
            None
        }
    }

    fn last(self) -> Option<Self::Item> {
// Check if already at the end
        if self.row_pos < self.slice_rows {
            unsafe {
// Get pointer to last row and create a slice from raw parts
                let ptr = self.slice_start.offset((self.slice_rows - 1) as isize * self.row_stride);
                Some(slice::$slice_from_parts(ptr, self.slice_cols))
            }
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.row_pos + n < self.slice_rows {
            let row: $row_type;
            unsafe {
                let ptr = self.slice_start.offset((self.row_pos + n) as isize * self.row_stride);
                row = slice::$slice_from_parts(ptr, self.slice_cols);
            }

            self.row_pos += n + 1;
            Some(row)
        } else {
            None
        }
    }

    fn count(self) -> usize {
        self.slice_rows - self.row_pos
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.slice_rows - self.row_pos, Some(self.slice_rows - self.row_pos))
    }
}
    );
);

impl_iter_rows!(Rows, &'a [T], from_raw_parts);
impl_iter_rows!(RowsMut, &'a mut [T], from_raw_parts_mut);

impl<'a, T> ExactSizeIterator for Rows<'a, T> {}
impl<'a, T> ExactSizeIterator for RowsMut<'a, T> {}

impl<T> Matrix<T> {
    /// Iterate over the rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    ///
    /// // Prints "2" three times.
    /// for row in a.iter_rows() {
    ///     println!("{}", row.len());
    /// }
    /// ```
    pub fn iter_rows(&self) -> Rows<T> {
        Rows {
            slice_start: self.data.as_ptr(),
            row_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.cols as isize,
            _marker: PhantomData::<&T>,
        }
    }

    /// Iterate over the mutable rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let mut a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    ///
    /// for row in a.iter_rows_mut() {
    ///     for r in row {
    ///         *r = *r + 1;
    ///     }
    /// }
    ///
    /// // Now contains the range 1..7
    /// println!("{}", a);
    /// ```
    pub fn iter_rows_mut(&mut self) -> RowsMut<T> {
        RowsMut {
            slice_start: self.data.as_mut_ptr(),
            row_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.cols as isize,
            _marker: PhantomData::<&mut T>,
        }
    }
}

impl<'a, T> MatrixSlice<'a, T> {
    /// Iterate over the rows of the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSlice};
    ///
    /// let a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    /// let b = MatrixSlice::from_matrix(&a, [0,0], 2, 2);
    ///
    /// // Prints "2" two times.
    /// for row in b.iter_rows() {
    ///     println!("{}", row.len());
    /// }
    /// ```
    pub fn iter_rows(&self) -> Rows<T> {
        Rows {
            slice_start: self.ptr,
            row_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride as isize,
            _marker: PhantomData::<&'a T>,
        }
    }
}

impl<'a, T> MatrixSliceMut<'a, T> {
    /// Iterate over the rows of the mutable matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    ///
    /// let mut a = Matrix::new(3 ,2, (0..6).collect::<Vec<usize>>());
    /// let b = MatrixSliceMut::from_matrix(&mut a, [0,0], 2, 2);
    ///
    /// // Prints "2" two times.
    /// for row in b.iter_rows() {
    ///     println!("{}", row.len());
    /// }
    /// ```
    pub fn iter_rows(&self) -> Rows<T> {
        Rows {
            slice_start: self.ptr,
            row_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride as isize,
            _marker: PhantomData::<&'a T>,
        }
    }

    /// Iterate over the mutable rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::{Matrix, MatrixSliceMut};
    ///
    /// let mut a = Matrix::new(3, 2, (0..6).collect::<Vec<usize>>());
    ///
    /// // New scope (so we can consume `a` after)
    /// {
    ///    let b = MatrixSliceMut::from_matrix(&mut a, [0,0], 2, 2);
    ///
    ///     for row in b.iter_rows_mut() {
    ///         for r in row {
    ///             *r = *r + 1;
    ///         }
    ///     }
    /// }
    ///
    /// // The first two rows have been incremented by 1
    /// println!("{}", a);
    /// ```
    pub fn iter_rows_mut(&self) -> RowsMut<T> {
        RowsMut {
            slice_start: self.ptr,
            row_pos: 0,
            slice_rows: self.rows,
            slice_cols: self.cols,
            row_stride: self.row_stride as isize,
            _marker: PhantomData::<&'a mut T>,
        }
    }
}

/// Creates a `Matrix` from an iterator over slices.
///
/// Each of the slices produced by the iterator will become a row in the matrix.
///
/// # Panics
///
/// Will panic if the iterators items do not have constant length.
///
/// # Examples
///
/// We can create a new matrix from some data.
///
/// ```
/// use rulinalg::matrix::Matrix;
///
/// let a : Matrix<f64> = vec![4f64; 16].chunks(4).collect();
///
/// assert_eq!(a.rows(), 4);
/// assert_eq!(a.cols(), 4);
/// ```
///
/// We can also do more interesting things.
///
/// ```
/// use rulinalg::matrix::Matrix;
///
/// let a = Matrix::new(4,2, (0..8).collect::<Vec<usize>>());
///
/// // Here we skip the first row and take only those
/// // where the first entry is less than 6.
/// let b = a.iter_rows()
///          .skip(1)
///          .filter(|x| x[0] < 6)
///          .collect::<Matrix<usize>>();
///
/// // We take the middle rows
/// assert_eq!(b.into_vec(), vec![2,3,4,5]);
/// ```
impl<'a, T: 'a + Copy> FromIterator<&'a [T]> for Matrix<T> {
    fn from_iter<I: IntoIterator<Item = &'a [T]>>(iterable: I) -> Self {
        let mut mat_data: Vec<T>;
        let cols: usize;
        let mut rows = 0;

        let mut iterator = iterable.into_iter();

        match iterator.next() {
            None => {
                return Matrix {
                    data: Vec::new(),
                    rows: 0,
                    cols: 0,
                }
            }
            Some(row) => {
                rows += 1;
                // Here we set the capacity - get iterator size and the cols
                let (lower_rows, _) = iterator.size_hint();
                cols = row.len();

                mat_data = Vec::with_capacity(lower_rows.saturating_add(1).saturating_mul(cols));
                mat_data.extend_from_slice(row);
            }
        }

        for row in iterator {
            assert!(row.len() == cols, "Iterator slice length must be constant.");
            mat_data.extend_from_slice(row);
            rows += 1;
        }

        mat_data.shrink_to_fit();

        Matrix {
            data: mat_data,
            rows: rows,
            cols: cols,
        }
    }
}

impl<'a, T> IntoIterator for MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut MatrixSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for MatrixSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a MatrixSliceMut<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut MatrixSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {

    use super::super::{Matrix, MatrixSlice, MatrixSliceMut};

    #[test]
    fn test_matrix_rows() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];

        for (i, row) in a.iter_rows().enumerate() {
            assert_eq!(data[i], *row);
        }

        for (i, row) in a.iter_rows_mut().enumerate() {
            assert_eq!(data[i], *row);
        }

        for row in a.iter_rows_mut() {
            for r in row {
                *r = 0;
            }
        }

        assert_eq!(a.into_vec(), vec![0; 9]);
    }

    #[test]
    fn test_matrix_slice_rows() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let b = MatrixSlice::from_matrix(&a, [0, 0], 2, 2);

        let data = [[0, 1], [3, 4]];

        for (i, row) in b.iter_rows().enumerate() {
            assert_eq!(data[i], *row);
        }
    }

    #[test]
    fn test_matrix_slice_mut_rows() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        {
            let b = MatrixSliceMut::from_matrix(&mut a, [0, 0], 2, 2);

            let data = [[0, 1], [3, 4]];

            for (i, row) in b.iter_rows().enumerate() {
                assert_eq!(data[i], *row);
            }

            for (i, row) in b.iter_rows_mut().enumerate() {
                assert_eq!(data[i], *row);
            }

            for row in b.iter_rows_mut() {
                for r in row {
                    *r = 0;
                }
            }
        }

        assert_eq!(a.into_vec(), vec![0, 0, 2, 0, 0, 5, 6, 7, 8]);
    }

    #[test]
    fn test_matrix_rows_nth() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let mut row_iter = a.iter_rows();

        assert_eq!([0, 1, 2], *row_iter.nth(0).unwrap());
        assert_eq!([6, 7, 8], *row_iter.nth(1).unwrap());

        assert_eq!(None, row_iter.next());
    }

    #[test]
    fn test_matrix_rows_last() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let row_iter = a.iter_rows();

        assert_eq!([6, 7, 8], *row_iter.last().unwrap());

        let mut row_iter = a.iter_rows();

        row_iter.next();
        assert_eq!([6, 7, 8], *row_iter.last().unwrap());

        let mut row_iter = a.iter_rows();

        row_iter.next();
        row_iter.next();
        row_iter.next();
        row_iter.next();

        assert_eq!(None, row_iter.last());
    }

    #[test]
    fn test_matrix_rows_count() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let row_iter = a.iter_rows();

        assert_eq!(3, row_iter.count());

        let mut row_iter_2 = a.iter_rows();
        row_iter_2.next();
        assert_eq!(2, row_iter_2.count());
    }

    #[test]
    fn test_matrix_rows_size_hint() {
        let a = Matrix::new(3, 3, (0..9).collect::<Vec<usize>>());

        let mut row_iter = a.iter_rows();

        assert_eq!((3, Some(3)), row_iter.size_hint());

        row_iter.next();

        assert_eq!((2, Some(2)), row_iter.size_hint());
        row_iter.next();
        row_iter.next();

        assert_eq!((0, Some(0)), row_iter.size_hint());

        assert_eq!(None, row_iter.next());
        assert_eq!((0, Some(0)), row_iter.size_hint());

    }

    #[test]
    fn into_iter_compile() { 
        let a = Matrix::new(3, 3, vec![2.0; 9]); 
        let mut b = MatrixSlice::from_matrix(&a, [1, 1], 2, 2);
    
        for _ in b { 
        } 
    
        for _ in &b { 
        } 
    
        for _ in &mut b { 
        } 
    } 
    
    #[test]
    fn into_iter_mut_compile() { 
        let mut a = Matrix::<f32>::new(3, 3, vec![2.0; 9]); 
        
        {
            let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
                
            for v in b { 
                *v = 1.0;
            } 
        }
    
        {
            let b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
    
            for _ in &b {
            } 
        }
    
        {
            let mut b = MatrixSliceMut::from_matrix(&mut a, [1, 1], 2, 2);
    
            for v in &mut b {
                *v = 1.0; 
            } 
        }
    } 
}
