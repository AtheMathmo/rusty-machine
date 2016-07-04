use super::{Matrix, MatrixSlice, MatrixSliceMut};
use super::slice::BaseSlice;

use std::any::{Any, TypeId};
use std::ops::{Add, Mul};

use libnum::Zero;
use matrixmultiply;

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
                            new_data[i*r + j] = *new_data.get_unchecked(i*r + j) +
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
impl<'a, 'b, T> Mul<&'a Matrix<T>> for Matrix<T>
    where T: Any + Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>
{
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, T> Mul<Matrix<T>> for &'a Matrix<T>
    where T: Any + Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>
{
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T>
    where T: Any + Copy + Zero + Add<T, Output = T> + Mul<T, Output = T>
{
    type Output = Matrix<T>;

    mat_mul_general!(Matrix);
}

macro_rules! impl_mat_slice_mul (
    ($slice:ident) => (

/// Multiplies two matrices together.
impl<'a, T> Mul<$slice<'a, T>> for Matrix<T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: $slice<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T> Mul<&'b $slice<'a, T>> for Matrix<T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: &$slice<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T> Mul<$slice<'a, T>> for &'b Matrix<T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: $slice<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T> Mul<&'c $slice<'a, T>> for &'b Matrix<T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    mat_mul_general!($slice);
}

/// Multiplies two matrices together.
impl<'a, T> Mul<Matrix<T>> for $slice<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T> Mul<&'b Matrix<T>> for $slice<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, T> Mul<Matrix<T>> for &'b $slice<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T> Mul<&'c Matrix<T>> for &'b $slice<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
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
impl<'a, 'b, T> Mul<$slice_2<'b, T>> for $slice_1<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: $slice_2<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T> Mul<&'c $slice_2<'b, T>> for $slice_1<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: &$slice_2<T>) -> Matrix<T> {
        (&self) * (m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, T> Mul<$slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    fn mul(self, m: $slice_2<T>) -> Matrix<T> {
        (self) * (&m)
    }
}

/// Multiplies two matrices together.
impl<'a, 'b, 'c, 'd, T> Mul<&'d $slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Any + Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>
{
    type Output = Matrix<T>;

    mat_mul_general!($slice_2);
}
    );
);

impl_slice_mul!(MatrixSlice, MatrixSlice);
impl_slice_mul!(MatrixSlice, MatrixSliceMut);
impl_slice_mul!(MatrixSliceMut, MatrixSlice);
impl_slice_mul!(MatrixSliceMut, MatrixSliceMut);

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
}
