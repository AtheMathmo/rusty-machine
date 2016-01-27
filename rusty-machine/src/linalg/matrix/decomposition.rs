use std::ops::{Mul, Add, Div, Sub, Neg};

use linalg::matrix::Matrix;
use linalg::vector::Vector;
use linalg::Metric;
use linalg::utils;

use libnum::{One, Zero, Float};

impl<T: Copy + Zero + Float> Matrix<T> {
    /// Cholesky decomposition
    ///
    /// Returns the cholesky decomposition of a positive definite matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let l = m.cholesky();
    /// ```
    ///
    /// # Panics
    ///
    /// - Matrix is not square.
    /// - Matrix is not positive definite. (This should probably be a Failure not a Panic).
    pub fn cholesky(&self) -> Matrix<T> {
        assert!(self.rows() == self.cols(), "Matrix is not square.");

        let mut new_data = Vec::<T>::with_capacity(self.rows() * self.cols());

        for i in 0..self.rows() {

            for j in 0..self.cols() {

                if j > i {
                    new_data.push(T::zero());
                    continue;
                }

                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + (new_data[i * self.cols() + k] * new_data[j * self.cols() + k]);
                }

                if j == i {
                    new_data.push((self[[i, i]] - sum).sqrt());
                } else {
                    let p = (self[[i, j]] - sum) / new_data[j * self.cols + j];

                    assert!(!p.is_nan(), "Matrix is not positive definite.");
                    new_data.push(p);
                }
            }
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: new_data,
        }
    }

    fn make_householder(mat: Matrix<T>) -> Matrix<T> {
        assert!(mat.cols() == 1usize, "Householder matrix has invalid size.");
        let size = mat.rows();

        let denom = mat.data()[0] + mat.data()[0].signum() * mat.norm();

        if denom == T::zero() {
            panic!("Matrix can not be decomposed.");
        }
        let mut v = (mat / denom).into_vec();
        v[0] = T::one();
        let v = Vector::new(v);
        let v_norm_sq = v.dot(&v);

        let v_vert = Matrix::new(size, 1, v.data().clone());
        let v_hor = Matrix::new(1, size, v.into_vec());
        Matrix::<T>::identity(size) - (v_vert * v_hor) * ((T::one() + T::one()) / v_norm_sq)


    }

    /// Compute the QR decomposition of the matrix.
    ///
    /// Returns the tuple (Q,R).
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let l = m.qr_decomp();
    /// ```
    pub fn qr_decomp(self) -> (Matrix<T>, Matrix<T>) {
        let m = self.rows();
        let n = self.cols();

        let mut q = Matrix::<T>::identity(m);
        let mut r = self;

        for i in 0..(n - ((m == n) as usize)) {
            let lower_rows = &(i..m).collect::<Vec<usize>>()[..];
            let lower_self = (r.select_cols(&[i])).select_rows(lower_rows);
            let mut holder_data = Matrix::make_householder(lower_self).into_vec();

            // This bit is inefficient
            // using for now as we'll swap to lapack eventually.
            let mut h_full_data = Vec::with_capacity(m * m);

            for j in 0..m {
                let mut row_data: Vec<T>;
                if j < i {
                    row_data = vec![T::zero(); m];
                    row_data[j] = T::one();
                    h_full_data.extend(row_data);
                } else {
                    row_data = vec![T::zero();i];
                    h_full_data.extend(row_data);
                    h_full_data.extend(holder_data.drain(..m - i));
                }
            }

            let h = Matrix::new(m, m, h_full_data);

            q = q * &h;
            r = h * &r;
        }

        (q, r)
    }
}


impl<T> Matrix<T> where T: Copy + One + Zero + Neg<Output=T> +
                           Add<T, Output=T> + Mul<T, Output=T> +
                           Sub<T, Output=T> + Div<T, Output=T> +
                           PartialOrd {

    /// Computes L, U, and P for LUP decomposition.
    ///
    /// Returns L,U, and P respectively.
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
    /// let (l,u,p) = a.lup_decomp();
    /// ```
    pub fn lup_decomp(&self) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        assert!(self.rows == self.cols, "Matrix is not square.");

        let n = self.cols;

        let mut l = Matrix::<T>::zeros(n, n);
        let mut u = Matrix::<T>::zeros(n, n);

        let mt = self.transpose();

        let mut p = Matrix::<T>::identity(n);

        // Compute the permutation matrix
        for i in 0..n {
            let (row,_) = utils::argmax(&mt.data[i*(n+1)..(i+1)*n]);

            if row != 0 {
                for j in 0..n {
                    p.data.swap(i*n + j, row*n+j)
                }
            }
        }

        let a_2 = &p * self;

        for i in 0..n {
            l.data[i*(n+1)] = T::one();

            for j in 0..i+1 {
                let mut s1 = T::zero();

                for k in 0..j {
                    s1 = s1 + l.data[j*n + k] * u.data[k*n + i];
                }

                u.data[j*n + i] = a_2[[j,i]] - s1;
            }

            for j in i..n {
                let mut s2 = T::zero();

                for k in 0..i {
                    s2 = s2 + l.data[j*n + k] * u.data[k*n + i];
                }

                let denom = u[[i,i]];

                if denom == T::zero() {
                    panic!("Arithmetic error. Matrix could not be decomposed.")
                }
                l.data[j*n + i] = (a_2[[j,i]] - s2) / denom;
            }

        }

        (l,u,p)
    }
}