use super::{Matrix, MatrixSlice, MatrixSliceMut, Axes};
use super::slice::BaseSlice;

use std::any::{Any, TypeId};
use std::ops::{Add, Mul};

use libnum::Zero;
use matrixmultiply;
use rayon;

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: Any, B: Any>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

macro_rules! mat_mul_general (
    ($mat:ident) => (
    
    fn mul(self, m: &$mat<T>) -> Matrix<T> {
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let p = self.rows;
        let q = self.cols;
        let r = m.cols;

        if same_type::<T, f32>() {
            let mut new_data = Vec::with_capacity(p * r);

            unsafe {
                new_data.set_len(p * r);

                matrixmultiply::sgemm(
                    p, q, r,
                    1f32,
                    self.as_ptr() as *const _,
                    self.row_stride() as isize, 1,
                    m.as_ptr() as *const _,
                    m.row_stride() as isize, 1,
                    0f32,
                    new_data.as_mut_ptr() as *mut _,
                    r as isize, 1
                    );
            }

            Matrix {
                rows: p,
                cols: r,
                data: new_data
            }
        } else if same_type::<T, f64>() {
            let mut new_data = Vec::with_capacity(p * r);

            unsafe {
                new_data.set_len(p * r);

                matrixmultiply::dgemm(
                    p, q, r,
                    1f64,
                    self.as_ptr() as *const _,
                    self.row_stride() as isize, 1,
                    m.as_ptr() as *const _,
                    m.row_stride() as isize, 1,
                    0f64,
                    new_data.as_mut_ptr() as *mut _,
                    r as isize, 1
                    );
            }

            Matrix {
                rows: p,
                cols: r,
                data: new_data
            }

        } else {
            let mut new_data = vec![T::zero(); p * r];

            unsafe {
                for i in 0..p
                {
                    for k in 0..q
                    {
                        for j in 0..r
                        {
                            new_data[i*r + j] = *new_data.get_unchecked(i*r + j) + *self.get_unchecked([i,k]) * *m.get_unchecked([k,j]);
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

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    mat_mul_general!(Matrix);
}

macro_rules! impl_mat_slice_mul (
    ($slice:ident) => (

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$slice<'a, T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $slice<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'b $slice<'a, T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$slice<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$slice<'a, T>> for &'b Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $slice<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'c $slice<'a, T>> for &'b Matrix<T> {
    type Output = Matrix<T>;

    mat_mul_general!($slice);
}

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<Matrix<T>> for $slice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'b Matrix<T>> for $slice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<Matrix<T>> for &'b $slice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'c Matrix<T>> for &'b $slice<'a, T> {
    type Output = Matrix<T>;

    mat_mul_general!(Matrix);
}
    );
);

impl_mat_slice_mul!(MatrixSlice);
impl_mat_slice_mul!(MatrixSliceMut);

macro_rules! impl_slice_mul (
    ($slice_1:ident, $slice_2:ident) => (

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$slice_2<'b, T>> for $slice_1<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: $slice_2<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'c $slice_2<'b, T>> for $slice_1<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$slice_2<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$slice_2<'b, T>> for &'c $slice_1<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: $slice_2<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, 'd,T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'d $slice_2<'b, T>> for &'c $slice_1<'a, T> {
    type Output = Matrix<T>;

    mat_mul_general!($slice_2);
}
    );
);

impl_slice_mul!(MatrixSlice, MatrixSlice);
impl_slice_mul!(MatrixSlice, MatrixSliceMut);
impl_slice_mul!(MatrixSliceMut, MatrixSlice);
impl_slice_mul!(MatrixSliceMut, MatrixSliceMut);

pub trait ParaMul
    : Any
    + Copy
    + Sync
    + Send
    + Zero
    + Mul<Self, Output=Self>
    + Add<Self, Output=Self>
{}

impl ParaMul for f32 {}
impl ParaMul for f64 {}

use std::marker::{Sync, Send};

unsafe impl<'a, T: Send> Send for MatrixSlice<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatrixSlice<'a, T> {}
unsafe impl<'a, T: Send> Send for MatrixSliceMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatrixSliceMut<'a, T> {}

fn fastmul<T: ParaMul>(a: &MatrixSlice<T>, b: &MatrixSlice<T>, c: MatrixSliceMut<T>) {
    let p = a.rows;
    let q = a.cols;
    let r = b.cols;

    assert!(q == b.rows, "Matrix dimensions do not agree.");
    assert!(p == c.rows);
    assert!(r == c.cols);

    if same_type::<T, f32>() {
        unsafe {
            matrixmultiply::sgemm(p,
                                  q,
                                  r,
                                  1f32,
                                  a.ptr as *const _,
                                  a.row_stride() as isize,
                                  1,
                                  b.ptr as *const _,
                                  b.row_stride() as isize,
                                  1,
                                  0f32,
                                  c.ptr as *mut _,
                                  c.row_stride as isize,
                                  1);
        }
    } else if same_type::<T, f64>() {
        unsafe {
            matrixmultiply::dgemm(p,
                                  q,
                                  r,
                                  1f64,
                                  a.ptr as *const _,
                                  a.row_stride() as isize,
                                  1,
                                  b.ptr as *const _,
                                  b.row_stride() as isize,
                                  1,
                                  0f64,
                                  c.ptr as *mut _,
                                  c.row_stride as isize,
                                  1);
        }
    }
}

impl<T: ParaMul> Matrix<T> {
    pub fn paramul(&self, m: &Matrix<T>) -> Matrix<T> {
        let s_1 = MatrixSlice::from_matrix(self, [0, 0], self.rows, self.cols);
        let s_2 = MatrixSlice::from_matrix(m, [0, 0], m.rows, m.cols);

        // Create uninitialized memory
        let mut t_vec = Vec::with_capacity(self.rows.saturating_mul(m.cols));
        unsafe {
            t_vec.set_len(self.rows.saturating_mul(m.cols));
        }

        // Create mat holding and fill slice
        let mut ret_mat = Matrix::new(self.rows, m.cols, t_vec);
        {
            let mat_slice = MatrixSliceMut::from_matrix(&mut ret_mat, [0, 0], self.rows, m.cols);

            s_1.paramul(&s_2, mat_slice);
        }

        ret_mat
    }
}

impl<'a, T: ParaMul> MatrixSlice<'a, T> {
    /// Multiplies matrices using Parallel divide and conquer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3,vec![3.0; 9]);
    /// let b = Matrix::new(3,3,vec![1.0; 9]);
    ///
    /// let c = a.paramul(&b);
    /// assert_eq!(c.into_vec(), vec![9.0; 9]);
    /// ```
    pub fn paramul(&self, m: &MatrixSlice<T>, mut c: MatrixSliceMut<T>) {
        let n = self.rows();
        let p = self.cols();
        let q = m.cols();

        let mut max_dim = n;

        if max_dim < p {
            max_dim = p;
        }
        if max_dim < q {
            max_dim = q;
        }

        if max_dim < 256 {
            fastmul(self, m, c);
        } else {
            let split_point = max_dim / 2;

            if max_dim == n {
                let (top, bottom) = self.split_at(split_point, Axes::Row);
                let (c_top, c_bottom) = c.split_at(split_point, Axes::Row);

                rayon::join(|| top.paramul(m, c_top), || bottom.paramul(m, c_bottom));
            } else if max_dim == p {
                // Split self vertically and b horizontally
                let (a_left, a_right) = self.split_at(split_point, Axes::Col);

                let (b_top, b_bottom) = m.split_at(split_point, Axes::Row);

                let mut t_vec = Vec::with_capacity(n.saturating_mul(q));
                unsafe {
                    t_vec.set_len(n.saturating_mul(q));
                }
                let mut t_mat = Matrix::new(n, q, t_vec);
                {
                    let t_mat_slice = MatrixSliceMut::from_matrix(&mut t_mat, [0, 0], n, q);

                    rayon::join(|| a_left.paramul(&b_top, c.clone()),
                                || a_right.paramul(&b_bottom, t_mat_slice));
                }

                c += t_mat
            } else if max_dim == q {
                // Split m vertically
                let (left, right) = m.split_at(split_point, Axes::Col);
                let (c_left, c_right) = c.split_at(split_point, Axes::Col);

                rayon::join(|| self.paramul(&left, c_left),
                            || self.paramul(&right, c_right));

            } else {
                panic!("Couldn't find the max of the matrix dimensions.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Matrix;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;

    #[test]
    fn matrix_mul_f32() {
        let a = Matrix::new(3, 2, vec![1f32, 2., 3., 4., 5., 6.]);
        let b = Matrix::new(2, 3, vec![1f32, 2., 3., 4., 5., 6.]);

        // Allocating new memory
        let c = &a * &b;

        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 3);

        assert_eq!(c[[0, 0]], 9.0);
        assert_eq!(c[[0, 1]], 12.0);
        assert_eq!(c[[0, 2]], 15.0);
        assert_eq!(c[[1, 0]], 19.0);
        assert_eq!(c[[1, 1]], 26.0);
        assert_eq!(c[[1, 2]], 33.0);
        assert_eq!(c[[2, 0]], 29.0);
        assert_eq!(c[[2, 1]], 40.0);
        assert_eq!(c[[2, 2]], 51.0);
    }

    #[test]
    fn matrix_mul_f64() {
        let a = Matrix::new(3, 2, vec![1f64, 2., 3., 4., 5., 6.]);
        let b = Matrix::new(2, 3, vec![1f64, 2., 3., 4., 5., 6.]);

        // Allocating new memory
        let c = &a * &b;

        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 3);

        assert_eq!(c[[0, 0]], 9.0);
        assert_eq!(c[[0, 1]], 12.0);
        assert_eq!(c[[0, 2]], 15.0);
        assert_eq!(c[[1, 0]], 19.0);
        assert_eq!(c[[1, 1]], 26.0);
        assert_eq!(c[[1, 2]], 33.0);
        assert_eq!(c[[2, 0]], 29.0);
        assert_eq!(c[[2, 1]], 40.0);
        assert_eq!(c[[2, 2]], 51.0);
    }

    #[test]
    fn matrix_mul_usize() {
        let a = Matrix::new(3, 2, vec![1usize, 2, 3, 4, 5, 6]);
        let b = Matrix::new(2, 3, vec![1usize, 2, 3, 4, 5, 6]);

        // Allocating new memory
        let c = &a * &b;

        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 3);

        assert_eq!(c[[0, 0]], 9);
        assert_eq!(c[[0, 1]], 12);
        assert_eq!(c[[0, 2]], 15);
        assert_eq!(c[[1, 0]], 19);
        assert_eq!(c[[1, 1]], 26);
        assert_eq!(c[[1, 2]], 33);
        assert_eq!(c[[2, 0]], 29);
        assert_eq!(c[[2, 1]], 40);
        assert_eq!(c[[2, 2]], 51);
    }

    #[test]
    fn mul_slice_basic() {
        let a = 3.0;
        let b = Matrix::new(2, 2, vec![1.0; 4]);
        let mut c = Matrix::new(3, 3, vec![2.0; 9]);
        {
            let d = MatrixSlice::from_matrix(&c, [1, 1], 2, 2);

            let m_1 = &d * a.clone();
            assert_eq!(m_1.into_vec(), vec![6.0; 4]);

            let m_2 = &d * b.clone();
            assert_eq!(m_2.into_vec(), vec![4.0; 4]);

            let m_3 = &d * &d;
            assert_eq!(m_3.into_vec(), vec![8.0; 4]);
        }

        let e = MatrixSliceMut::from_matrix(&mut c, [1, 1], 2, 2);

        let m_1 = &e * a;
        assert_eq!(m_1.into_vec(), vec![6.0; 4]);

        let m_2 = &e * b;
        assert_eq!(m_2.into_vec(), vec![4.0; 4]);

        let m_3 = &e * &e;
        assert_eq!(m_3.into_vec(), vec![8.0; 4]);
    }

    #[test]
    fn mul_slice_uneven_data() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let c = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d = MatrixSlice::from_matrix(&c, [0, 0], 2, 2);

        let e = d * a;

        assert_eq!(e[[0, 0]], 7.0);
        assert_eq!(e[[0, 1]], 10.0);
        assert_eq!(e[[1, 0]], 19.0);
        assert_eq!(e[[1, 1]], 28.0);
    }

    #[test]
    fn mul_slice_uneven_data_usize() {
        let a = Matrix::new(2, 2, vec![1usize, 2, 3, 4]);

        let c = Matrix::new(2, 3, vec![1usize, 2, 3, 4, 5, 6]);
        let d = MatrixSlice::from_matrix(&c, [0, 0], 2, 2);

        let e = d * a;

        assert_eq!(e[[0, 0]], 7);
        assert_eq!(e[[0, 1]], 10);
        assert_eq!(e[[1, 0]], 19);
        assert_eq!(e[[1, 1]], 28);
    }

    #[test]
    fn paramul_n_large() {
        let a = Matrix::new(1000, 20, vec![2.0; 20000]);
        let b = Matrix::new(20, 20, vec![2.0; 400]);

        let c = a.paramul(&b);

        assert_eq!(c.into_vec(), vec![80.0; 20000]);
    }

    #[test]
    fn paramul_k_large() {
        let a = Matrix::new(20, 1000, vec![2.0; 20000]);
        let b = Matrix::new(1000, 20, vec![2.0; 20000]);

        let c = a.paramul(&b);

        assert_eq!(c.into_vec(), vec![4000.0; 400]);
    }

    #[test]
    fn paramul_m_large() {
        let a = Matrix::new(20, 20, vec![2.0; 400]);
        let b = Matrix::new(20, 1000, vec![2.0; 20000]);

        let c = a.paramul(&b);

        assert_eq!(c.into_vec(), vec![80.0; 20000]);
    }
}
