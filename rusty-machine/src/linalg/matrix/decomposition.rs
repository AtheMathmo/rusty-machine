//! Matrix Decompositions
//!
//! References:
//! 1. [On Matrix Balancing and EigenVector computation](http://arxiv.org/pdf/1401.5766v1.pdf), James, Langou and Lowery

use std::ops::{Mul, Add, Div, Sub, Neg};
use std::cmp;
use std::any::Any;

use linalg::matrix::Matrix;
use linalg::vector::Vector;
use linalg::Metric;
use linalg::utils;

use libnum::{One, Zero, Float, NumCast, Signed};
use libnum::{cast, abs};

impl<T: Any  + Float> Matrix<T> {
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

    fn make_householder_vec(mat: Matrix<T>) -> Matrix<T> {
        assert!(mat.cols() == 1usize, "Householder matrix has invalid size.");
        let size = mat.rows();

        let denom = mat.data()[0] + mat.data()[0].signum() * mat.norm();

        if denom == T::zero() {
            panic!("Matrix can not be decomposed.");
        }

        let mut v = (mat / denom).into_vec();
        v[0] = T::one();
        let v = Matrix::new(size, 1, v);

        &v / v.norm()
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
            let lower_self = r.select(lower_rows, &[i]);
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

impl<T: Any + Float + Signed> Matrix<T> {
    /// Returns H, where H is the upper hessenberg form.
    ///
    /// If the transformation matrix is also required, you should
    /// use `upper_hess_decomp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(4,4,vec![2.,0.,1.,1.,2.,0.,1.,2.,1.,2.,0.,0.,2.,0.,1.,1.]);
    /// let h = a.upper_hessenberg();
    ///
    /// println!("{:?}", h.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn upper_hessenberg(&self) -> Matrix<T> {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        let mut dummy = self.clone();

        for i in 0..n - 2 {
            let lower_rows = &(i + 1..n).collect::<Vec<usize>>()[..];
            let lower_self = dummy.select(lower_rows, &[i]);;
            let h_holder_vec = Matrix::make_householder_vec(lower_self);

            let i_plus_to_n = (i + 1..n).collect::<Vec<usize>>();

            // Apply holder on the left
            let dummy_block = dummy.select(&i_plus_to_n, &(i..n).collect::<Vec<usize>>());
            let reduc_block = &dummy_block -
                              &h_holder_vec * (h_holder_vec.transpose() * &dummy_block) *
                              (T::one() + T::one());

            // Reassign block
            for j in i + 1..n {
                for k in i..n {
                    dummy.data[j * dummy.cols + k] = reduc_block.data[(j - i - 1) *
                                                                      reduc_block.cols +
                                                                      k -
                                                                      i]
                }
            }

            // Apply holder on the right
            let dummy_block = dummy.select(&(0..n).collect::<Vec<usize>>(), &i_plus_to_n);
            let reduc_block = &dummy_block -
                              (&dummy_block * &h_holder_vec) * h_holder_vec.transpose() *
                              (T::one() + T::one());

            // Reassign block
            for j in 0..n {
                for k in i + 1..n {
                    dummy.data[j * dummy.cols + k] = reduc_block.data[j * reduc_block.cols + k - i -
                                                                      1]
                }
            }

        }

        // Enforce upper hessenberg
        for i in 0..self.cols - 2 {
            for j in i + 2..self.rows {
                dummy.data[j * self.cols + i] = T::zero();
            }
        }

        dummy
    }

    /// Returns (U,H), where H is the upper hessenberg form
    /// and U is the unitary transform matrix.
    ///
    /// Note: The current transform matrix seems broken...
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3,vec![1.,2.,3.,4.,5.,6.,7.,8.,9.]);
    ///
    /// // u is the transform, h is the upper hessenberg form.
    /// let (u,h) = a.upper_hess_decomp();
    ///
    /// println!("The hess : {:?}", h.data());
    /// println!("Manual hess : {:?}", (u.transpose() * &a * u).data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn upper_hess_decomp(&self) -> (Matrix<T>, Matrix<T>) {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        // First we form the transformation.
        let mut transform = Matrix::identity(n);

        for i in (0..n - 2).rev() {
            let lower_rows = (i + 1..n).collect::<Vec<usize>>();
            let lower_self = self.select(&lower_rows, &[i]);
            let h_holder_vec = Matrix::make_householder_vec(lower_self);

            let trans_block = transform.select(&lower_rows, &lower_rows);
            let reduc_block = &trans_block -
                              &h_holder_vec * (h_holder_vec.transpose() * &trans_block) *
                              (T::one() + T::one());

            // Reassign block
            for j in i + 1..n {
                for k in i + 1..n {
                    transform.data[j * n + k] = reduc_block.data[(j - i - 1) * reduc_block.cols +
                                                                 k -
                                                                 i -
                                                                 1];
                }
            }
        }

        // Now we reduce to upper hessenberg
        (transform, self.upper_hessenberg())
    }

    fn balance_matrix(&mut self) {
        let n = self.rows();
        let radix = T::one() + T::one();

        assert!(n == self.cols(),
                "Matrix must be square to produce balance matrix.");

        let mut d = Matrix::<T>::identity(n);
        let mut converged = false;

        while !converged {
            converged = true;

            for i in 0..n {
                let mut c = self.select_cols(&[i]).norm();
                let mut r = self.select_rows(&[i]).norm();

                let s = c * c + r * r;
                let mut f = T::one();

                while c < r / radix {
                    c = c * radix;
                    r = r / radix;
                    f = f * radix;
                }

                while c >= r * radix {
                    c = c / radix;
                    r = r * radix;
                    f = f / radix;
                }

                if (c * c + r * r) < cast::<f64, T>(0.95).unwrap() * s {
                    converged = false;
                    d.data[i * (self.cols + 1)] = f * d.data[i * (self.cols + 1)];

                    for j in 0..n {
                        self.data[j * self.cols + i] = f * self.data[j * self.cols + i];
                        self.data[i * self.cols + j] = self.data[i * self.cols + j] / f;
                    }
                }
            }
        }
    }

    /// Compute the cos and sin values for the givens rotation.
    ///
    /// Returns a tuple (c,s).
    fn givens_rot(a: T, b: T) -> (T, T) {
        let r = a.hypot(b);

        (a / r, -b / r)
    }

    /// Eigenvalues of a square matrix.
    ///
    /// Returns a Vec of eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// 
    /// let a = Matrix::new(4,4, (1..17).map(|v| v as f64).collect::<Vec<f64>>());
    /// let e = a.eigenvalues();
    /// println!("{:?}", e);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn eigenvalues(&self) -> Vec<T> {
        let n = self.rows();
        assert!(n == self.cols(), "Matrix must be square for eigendecomp.");

        let mut h = self.upper_hessenberg();
        h.balance_matrix();

        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).unwrap();

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = Matrix::make_householder(Matrix::new(3, 1, vec![x, y, z]));

                // Apply householder transformation to block (on the left)
                let h_block = h.select(&[k, k + 1, k + 2], &(r..n).collect::<Vec<usize>>());
                let reduc_block = &householder * h_block;

                // Reassign the block
                for i in k..k + 3 {
                    for j in r..n {
                        h.data[i * h.cols + j] = reduc_block.data[(i - k) * reduc_block.cols() + j -
                                                                  r];
                    }
                }

                let r = cmp::min(k + 4, p + 1);

                // Apply householder transformation to the block (on the right)
                let h_block = h.select(&(0..r).collect::<Vec<usize>>(), &[k, k + 1, k + 2]);
                let reduc_block = h_block * householder.transpose();

                // Reassign the block
                for i in 0..r {
                    for j in k..k + 3 {
                        h.data[i * h.cols + j] = reduc_block.data[i * reduc_block.cols + j - k];
                    }
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            // Apply Givens rotation to the block (on the left)
            let h_block = h.select(&(q..p + 1).collect::<Vec<usize>>(),
                                   &(p - 2..n).collect::<Vec<usize>>());
            let reduc_block = &givens_mat * h_block;

            // Reassign the block
            for i in q..p + 1 {
                for j in p - 2..n {
                    h.data[i * h.cols + j] = reduc_block.data[(i - q) * reduc_block.cols + j -
                                                              (p - 2)];
                }
            }

            // Apply Givens rotation to block (on the right)
            let h_block = h.select(&(0..p).collect::<Vec<usize>>(),
                                   &(p - 1..p + 1).collect::<Vec<usize>>());
            let reduc_block = h_block * givens_mat.transpose();

            // Reassign the block
            for i in 1..p {
                for j in p - 1..p + 1 {
                    h.data[i * h.cols + j] = reduc_block.data[i * reduc_block.cols + j - (p - 1)];
                }
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p = p - 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p = p - 2;
            }
        }

        h.diag().into_vec()
    }

    /// Eigen decomposition of a square matrix.
    ///
    /// Returns a Vec of eigenvalues, and a matrix with eigenvectors as the columns.
    ///
    /// The eigenvectors are only gauranteed to be correct if the matrix is real-symmetric.
    /// I'm not planning on doing any more work on this for now and will prefer to switch
    /// out to the lapack implementations in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// 
    /// let a = Matrix::new(3,3,vec![3.,2.,4.,2.,0.,2.,4.,2.,3.]);
    ///
    /// let (e, m) = a.eigendecomp();
    /// println!("{:?}", e);
    /// println!("{:?}", m.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    pub fn eigendecomp(&self) -> (Vec<T>, Matrix<T>) {
        let n = self.rows();
        assert!(n == self.cols(), "Matrix must be square for eigendecomp.");

        let (u, mut h) = self.upper_hess_decomp();
        h.balance_matrix();
        let mut transformation = Matrix::identity(n);

        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).unwrap();

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = Matrix::make_householder(Matrix::new(3, 1, vec![x, y, z]));

                // Apply householder transformation to block (on the left)
                let h_block = h.select(&[k, k + 1, k + 2], &(r..n).collect::<Vec<usize>>());
                let reduc_block = &householder * h_block;

                // Reassign the block
                for i in k..k + 3 {
                    for j in r..n {
                        h.data[i * h.cols + j] = reduc_block.data[(i - k) * reduc_block.cols() + j -
                                                                  r];
                    }
                }

                let r = cmp::min(k + 4, p + 1);

                // Apply householder transformation to the block (on the right)
                let h_block = h.select(&(0..r).collect::<Vec<usize>>(), &[k, k + 1, k + 2]);
                let reduc_block = h_block * householder.transpose();

                // Reassign the block
                for i in 0..r {
                    for j in k..k + 3 {
                        h.data[i * h.cols + j] = reduc_block.data[i * reduc_block.cols + j - k];
                    }
                }

                // Update the transformation matrix
                let trans_block = transformation.select(&(0..n).collect::<Vec<usize>>(),
                                                        &[k, k + 1, k + 2]);
                let reduc_block = trans_block * householder.transpose();

                // Reassign the trans block
                for i in 0..n {
                    for j in k..k + 3 {
                        transformation.data[i * transformation.cols + j] =
                            reduc_block.data[i * reduc_block.cols + j - k];
                    }
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            // Apply Givens rotation to the block (on the left)
            let h_block = h.select(&(q..p + 1).collect::<Vec<usize>>(),
                                   &(p - 2..n).collect::<Vec<usize>>());
            let reduc_block = &givens_mat * h_block;

            // Reassign the block
            for i in q..p + 1 {
                for j in p - 2..n {
                    h.data[i * h.cols + j] = reduc_block.data[(i - q) * reduc_block.cols + j -
                                                              (p - 2)];
                }
            }

            // Apply Givens rotation to block (on the right)
            let h_block = h.select(&(0..p).collect::<Vec<usize>>(),
                                   &(p - 1..p + 1).collect::<Vec<usize>>());
            let reduc_block = h_block * givens_mat.transpose();

            // Reassign the block
            for i in 1..p {
                for j in p - 1..p + 1 {
                    h.data[i * h.cols + j] = reduc_block.data[i * reduc_block.cols + j - (p - 1)];
                }
            }

            // Update the transformation matrix
            let trans_block = transformation.select(&(0..n).collect::<Vec<usize>>(), &[p - 1, p]);
            let reduc_block = trans_block * givens_mat.transpose();

            // Reassign the trans block
            for i in 0..n {
                for j in p - 1..p + 1 {
                    transformation.data[i * transformation.cols + j] =
                        reduc_block.data[i * reduc_block.cols + j - (p - 1)];
                }
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p = p - 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p = p - 2;
            }
        }

        (h.diag().into_vec(), u * transformation)
    }
}


impl<T> Matrix<T> where T: Any + Copy + One + Zero + Neg<Output=T> +
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
