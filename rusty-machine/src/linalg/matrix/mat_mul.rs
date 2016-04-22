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

macro_rules! impl_mat_mul (
    ($mat_1:ident, $mat_2:ident) => (

/// Multiplies two matrices together.
impl<T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$mat_2<T>> for $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $mat_2<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'a $mat_2<T>> for $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$mat_2<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<$mat_2<T>> for &'a $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: $mat_2<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T: Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>> Mul<&'b $mat_2<T>> for &'a $mat_1<T> {
    type Output = Matrix<T>;

    fn mul(self, m: &$mat_2<T>) -> Matrix<T> {
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
}
    );
);

impl_mat_mul!(Matrix, Matrix);
impl_mat_mul!(Matrix, MatrixSlice);
impl_mat_mul!(Matrix, MatrixSliceMut);
impl_mat_mul!(MatrixSlice, Matrix);
impl_mat_mul!(MatrixSlice, MatrixSlice);
impl_mat_mul!(MatrixSlice, MatrixSliceMut);
impl_mat_mul!(MatrixSliceMut, Matrix);
impl_mat_mul!(MatrixSliceMut, MatrixSlice);
impl_mat_mul!(MatrixSliceMut, MatrixSliceMut);

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

unsafe impl<T: Send> Send for MatrixSlice<T> {}
unsafe impl<T: Sync> Sync for MatrixSlice<T> {}

impl< T: ParaMul> Matrix<T> {
    pub fn paramul(&self, m: &Matrix<T>) -> Matrix<T> {
        let s_1 = MatrixSlice::from_matrix(self, [0,0], self.rows, self.cols);
        let s_2 = MatrixSlice::from_matrix(m, [0,0], m.rows, m.cols);

        s_1.paramul(&s_2)
    }
}

impl <T: ParaMul> MatrixSlice<T> {
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
    pub fn paramul(&self, m: &MatrixSlice<T>) -> Matrix<T> {
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
            self * m
        } else {
            let split_point = max_dim / 2;
            
            if max_dim == n {
                let (top, bottom) = self.split_at(split_point, Axes::Row);

                let (a_1_b, a_2_b) = rayon::join(|| top.paramul(m),
                                                 || bottom.paramul(m));

                a_1_b.vcat(&a_2_b)
            } else if max_dim == p {
                // Split self vertically and b horizontally
                let (a_left, a_right) = self.split_at(split_point, Axes::Col);

                let (b_top, b_bottom) = m.split_at(split_point, Axes::Row);

                let (a_1_b_1, a_2_b_2) = rayon::join(|| a_left.paramul(&b_top),
                                                     || a_right.paramul(&b_bottom));

                a_1_b_1 + a_2_b_2
            } else if max_dim == q {
                // Split m vertically
                let (left, right) = m.split_at(split_point, Axes::Col);

                let (a_b_1, a_b_2) = rayon::join(|| self.paramul(&left),
                                                 || self.paramul(&right));

                a_b_1.hcat(&a_b_2)
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

        let d = MatrixSlice::from_matrix(&c, [1, 1], 2, 2);

        let m_1 = &d * a.clone();
        assert_eq!(m_1.into_vec(), vec![6.0; 4]);

        let m_2 = &d * b.clone();
        assert_eq!(m_2.into_vec(), vec![4.0; 4]);

        let m_3 = &d * &d;
        assert_eq!(m_3.into_vec(), vec![8.0; 4]);

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

        assert_eq!(e[[0,0]], 7.0);
        assert_eq!(e[[0,1]], 10.0);
        assert_eq!(e[[1,0]], 19.0);
        assert_eq!(e[[1,1]], 28.0);
    }

    #[test]
    fn mul_slice_uneven_data_usize() {
        let a = Matrix::new(2, 2, vec![1usize, 2, 3, 4]);

        let c = Matrix::new(2, 3, vec![1usize, 2, 3, 4, 5, 6]);
        let d = MatrixSlice::from_matrix(&c, [0, 0], 2, 2);

        let e = d * a;

        assert_eq!(e[[0,0]], 7);
        assert_eq!(e[[0,1]], 10);
        assert_eq!(e[[1,0]], 19);
        assert_eq!(e[[1,1]], 28);
    }
}
