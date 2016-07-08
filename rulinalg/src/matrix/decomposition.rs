//! Matrix Decompositions
//!
//! References:
//! 1. [On Matrix Balancing and EigenVector computation]
//! (http://arxiv.org/pdf/1401.5766v1.pdf), James, Langou and Lowery

use std::ops::{Mul, Add, Div, Sub, Neg};
use std::cmp;
use std::any::Any;

use matrix::{Matrix, MatrixSlice, MatrixSliceMut};
use vector::Vector;
use Metric;
use utils;
use error::{Error, ErrorKind};

use libnum::{One, Zero, Float, Signed};
use libnum::{cast, abs};

impl<T: Any + Float> Matrix<T> {
    /// Cholesky decomposition
    ///
    /// Returns the cholesky decomposition of a positive definite matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let l = m.cholesky();
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - Matrix is not positive definite.
    pub fn cholesky(&self) -> Result<Matrix<T>, Error> {
        assert!(self.rows == self.cols,
                "Matrix must be square for Cholesky decomposition.");

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

                    if !p.is_finite() {
                        return Err(Error::new(ErrorKind::DecompFailure,
                                              "Matrix is not positive definite."));
                    } else {

                    }
                    new_data.push(p);
                }
            }
        }

        Ok(Matrix {
            rows: self.rows(),
            cols: self.cols(),
            data: new_data,
        })
    }

    fn make_householder(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Vector::new(v);
        let v_norm_sq = v.dot(&v);

        let v_vert = Matrix::new(size, 1, v.data().clone());
        let v_hor = Matrix::new(1, size, v.into_vec());
        Ok(Matrix::<T>::identity(size) - (v_vert * v_hor) * ((T::one() + T::one()) / v_norm_sq))
    }

    fn make_householder_vec(column: &[T]) -> Result<Matrix<T>, Error> {
        let size = column.len();

        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidArg,
                                  "Column for householder transform cannot be empty."));
        }

        let denom = column[0] + column[0].signum() * utils::dot(column, column).sqrt();

        if denom == T::zero() {
            return Err(Error::new(ErrorKind::DecompFailure,
                                  "Cannot produce househoulder transform from column as first \
                                   entry is 0."));
        }

        let mut v = column.into_iter().map(|&x| x / denom).collect::<Vec<T>>();
        // Ensure first element is fixed to 1.
        v[0] = T::one();
        let v = Matrix::new(size, 1, v);

        Ok(&v / v.norm())
    }

    /// Compute the QR decomposition of the matrix.
    ///
    /// Returns the tuple (Q,R).
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let m = Matrix::new(3,3, vec![1.0,0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0]);
    ///
    /// let (q, r) = m.qr_decomp().unwrap();
    /// ```
    ///
    /// # Failures
    ///
    /// - Cannot compute the QR decomposition.
    pub fn qr_decomp(self) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let m = self.rows();
        let n = self.cols();

        let mut q = Matrix::<T>::identity(m);
        let mut r = self;

        for i in 0..(n - ((m == n) as usize)) {
            let holder_transform: Result<Matrix<T>, Error>;
            {
                let lower_slice = MatrixSlice::from_matrix(&r, [i, i], m - i, 1);
                holder_transform =
                    Matrix::make_householder(&lower_slice.iter().cloned().collect::<Vec<_>>());
            }

            if !holder_transform.is_ok() {
                return Err(Error::new(ErrorKind::DecompFailure,
                                      "Cannot compute QR decomposition."));
            } else {
                let mut holder_data = holder_transform.unwrap().into_vec();

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
                        row_data = vec![T::zero(); i];
                        h_full_data.extend(row_data);
                        h_full_data.extend(holder_data.drain(..m - i));
                    }
                }

                let h = Matrix::new(m, m, h_full_data);

                q = q * &h;
                r = h * &r;
            }
        }

        Ok((q, r))
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
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(4,4,vec![2.,0.,1.,1.,2.,0.,1.,2.,1.,2.,0.,0.,2.,0.,1.,1.]);
    /// let h = a.upper_hessenberg();
    ///
    /// println!("{:?}", h.expect("Could not get upper Hessenberg form.").data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
    pub fn upper_hessenberg(&self) -> Result<Matrix<T>, Error> {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        let mut dummy = self.clone();

        for i in 0..n - 2 {
            let h_holder_vec: Matrix<T>;
            {
                let lower_slice = MatrixSlice::from_matrix(&dummy, [i + 1, i], n - i - 1, 1);
                // Try to get the house holder transform - else map error and pass up.
                h_holder_vec = try!(Matrix::make_householder_vec(&lower_slice.iter()
                        .cloned()
                        .collect::<Vec<_>>())
                    .map_err(|_| {
                        Error::new(ErrorKind::DecompFailure,
                                   "Cannot compute upper Hessenberg form.")
                    }));
            }

            {
                // Apply holder on the left
                let mut dummy_block =
                    MatrixSliceMut::from_matrix(&mut dummy, [i + 1, i], n - i - 1, n - i);
                dummy_block -= &h_holder_vec * (h_holder_vec.transpose() * &dummy_block) *
                               (T::one() + T::one());
            }

            {
                // Apply holder on the right
                let mut dummy_block =
                    MatrixSliceMut::from_matrix(&mut dummy, [0, i + 1], n, n - i - 1);
                dummy_block -= (&dummy_block * &h_holder_vec) * h_holder_vec.transpose() *
                               (T::one() + T::one());
            }

        }

        // Enforce upper hessenberg
        for i in 0..self.cols - 2 {
            for j in i + 2..self.rows {
                dummy.data[j * self.cols + i] = T::zero();
            }
        }

        Ok(dummy)
    }

    /// Returns (U,H), where H is the upper hessenberg form
    /// and U is the unitary transform matrix.
    ///
    /// Note: The current transform matrix seems broken...
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3,vec![1.,2.,3.,4.,5.,6.,7.,8.,9.]);
    ///
    /// // u is the transform, h is the upper hessenberg form.
    /// let (u,h) = a.upper_hess_decomp().expect("This matrix should decompose!");
    ///
    /// println!("The hess : {:?}", h.data());
    /// println!("Manual hess : {:?}", (u.transpose() * &a * u).data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The matrix cannot be reduced to upper hessenberg form.
    pub fn upper_hess_decomp(&self) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let n = self.rows;
        assert!(n == self.cols,
                "Matrix must be square to produce upper hessenberg.");

        // First we form the transformation.
        let mut transform = Matrix::identity(n);

        for i in (0..n - 2).rev() {
            let h_holder_vec: Matrix<T>;
            {
                let lower_slice = MatrixSlice::from_matrix(&self, [i + 1, i], n - i - 1, 1);
                h_holder_vec = try!(Matrix::make_householder_vec(&lower_slice.iter()
                        .cloned()
                        .collect::<Vec<_>>())
                    .map_err(|_| {
                        Error::new(ErrorKind::DecompFailure,
                                   "Cannot compute upper Hessenberg decomposition.")
                    }));
            }

            let mut trans_block =
                MatrixSliceMut::from_matrix(&mut transform, [i + 1, i + 1], n - i - 1, n - i - 1);
            trans_block -= &h_holder_vec * (h_holder_vec.transpose() * &trans_block) *
                           (T::one() + T::one());
        }

        // Now we reduce to upper hessenberg
        Ok((transform, try!(self.upper_hessenberg())))
    }

    fn balance_matrix(&mut self) {
        let n = self.rows();
        let radix = T::one() + T::one();

        debug_assert!(n == self.cols(),
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

    fn direct_2_by_2_eigenvalues(&self) -> Result<Vec<T>, Error> {
        // The characteristic polynomial of a 2x2 matrix A is
        // λ² − (a₁₁ + a₂₂)λ + (a₁₁a₂₂ − a₁₂a₂₁);
        // the quadratic formula suffices.
        let tr = self.data[0] + self.data[3];
        let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];

        let two = T::one() + T::one();
        let four = two + two;

        let discr = tr * tr - four * det;

        if discr < T::zero() {
            Err(Error::new(ErrorKind::DecompFailure,
                           "Matrix has complex eigenvalues. Currently unsupported, sorry!"))
        } else {
            let discr_root = discr.sqrt();
            Ok(vec![(tr - discr_root) / two, (tr + discr_root) / two])
        }

    }

    fn francis_shift_eigenvalues(&self) -> Result<Vec<T>, Error> {
        let n = self.rows();
        debug_assert!(n > 2,
                      "Francis shift only works on matrices greater than 2x2.");
        debug_assert!(n == self.cols, "Matrix must be square for Francis shift.");

        let mut h = try!(self.upper_hessenberg()
            .map_err(|_| Error::new(ErrorKind::DecompFailure, "Could not compute eigenvalues.")));
        h.balance_matrix();

        // The final index of the active matrix
        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).expect("Failed to cast value for convergence check.");

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = try!(Matrix::make_householder(&[x, y, z]).map_err(|_| {
                    Error::new(ErrorKind::DecompFailure, "Could not compute eigenvalues.")
                }));

                {
                    // Apply householder transformation to block (on the left)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [k, r], 3, n - r);
                    let transformed = &householder * &h_block;
                    h_block.set_to(transformed.as_slice());
                }

                let r = cmp::min(k + 4, p + 1);

                {
                    // Apply householder transformation to the block (on the right)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [0, k], r, 3);
                    let transformed = &h_block * householder.transpose();
                    h_block.set_to(transformed.as_slice());
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            {
                // Apply Givens rotation to the block (on the left)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [q, p - 2], 2, n - p + 2);
                let transformed = &givens_mat * &h_block;
                h_block.set_to(transformed.as_slice());
            }

            {
                // Apply Givens rotation to block (on the right)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [0, q], p + 1, 2);
                let transformed = &h_block * givens_mat.transpose();
                h_block.set_to(transformed.as_slice());
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p -= 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p -= 2;
            }
        }

        Ok(h.diag().into_vec())
    }

    /// Eigenvalues of a square matrix.
    ///
    /// Returns a Vec of eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(4,4, (1..17).map(|v| v as f64).collect::<Vec<f64>>());
    /// let e = a.eigenvalues().expect("We should be able to compute these eigenvalues!");
    /// println!("{:?}", e);
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - Eigenvalues cannot be computed.
    pub fn eigenvalues(&self) -> Result<Vec<T>, Error> {
        let n = self.rows();
        assert!(n == self.cols,
                "Matrix must be square for eigenvalue computation.");

        match n {
            1 => Ok(vec![self.data[0]]),
            2 => self.direct_2_by_2_eigenvalues(),
            _ => self.francis_shift_eigenvalues(),
        }
    }

    fn direct_2_by_2_eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let eigenvalues = try!(self.eigenvalues());
        // Thanks to
        // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
        // for this characterization—
        if self.data[2] != T::zero() {
            let decomp_data = vec![eigenvalues[0] - self.data[3],
                                   eigenvalues[1] - self.data[3],
                                   self.data[2],
                                   self.data[2]];
            Ok((eigenvalues, Matrix::new(2, 2, decomp_data)))
        } else if self.data[1] != T::zero() {
            let decomp_data = vec![self.data[1],
                                   self.data[1],
                                   eigenvalues[0] - self.data[0],
                                   eigenvalues[1] - self.data[0]];
            Ok((eigenvalues, Matrix::new(2, 2, decomp_data)))
        } else {
            Ok((eigenvalues, Matrix::new(2, 2, vec![T::one(), T::zero(), T::zero(), T::one()])))
        }
    }

    fn francis_shift_eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let n = self.rows();
        debug_assert!(n > 2,
                      "Francis shift only works on matrices greater than 2x2.");
        debug_assert!(n == self.cols, "Matrix must be square for Francis shift.");

        let (u, mut h) = try!(self.upper_hess_decomp().map_err(|_| {
            Error::new(ErrorKind::DecompFailure,
                       "Could not compute eigen decomposition.")
        }));
        h.balance_matrix();
        let mut transformation = Matrix::identity(n);

        // The final index of the active matrix
        let mut p = n - 1;

        let eps = cast::<f64, T>(1e-20).expect("Failed to cast value for convergence check.");

        while p > 1 {
            let q = p - 1;
            let s = h[[q, q]] + h[[p, p]];
            let t = h[[q, q]] * h[[p, p]] - h[[q, p]] * h[[p, q]];

            let mut x = h[[0, 0]] * h[[0, 0]] + h[[0, 1]] * h[[1, 0]] - h[[0, 0]] * s + t;
            let mut y = h[[1, 0]] * (h[[0, 0]] + h[[1, 1]] - s);
            let mut z = h[[1, 0]] * h[[2, 1]];

            for k in 0..p - 1 {
                let r = cmp::max(1, k) - 1;

                let householder = try!(Matrix::make_householder(&[x, y, z]).map_err(|_| {
                    Error::new(ErrorKind::DecompFailure,
                               "Could not compute eigen decomposition.")
                }));

                {
                    // Apply householder transformation to block (on the left)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [k, r], 3, n - r);
                    let transformed = &householder * &h_block;
                    h_block.set_to(transformed.as_slice());
                }

                let r = cmp::min(k + 4, p + 1);

                {
                    // Apply householder transformation to the block (on the right)
                    let h_block = MatrixSliceMut::from_matrix(&mut h, [0, k], r, 3);
                    let transformed = &h_block * householder.transpose();
                    h_block.set_to(transformed.as_slice());
                }

                {
                    // Update the transformation matrix
                    let trans_block =
                        MatrixSliceMut::from_matrix(&mut transformation, [0, k], n, 3);
                    let transformed = &trans_block * householder.transpose();
                    trans_block.set_to(transformed.as_slice());
                }

                x = h[[k + 1, k]];
                y = h[[k + 2, k]];

                if k < p - 2 {
                    z = h[[k + 3, k]];
                }
            }

            let (c, s) = Matrix::givens_rot(x, y);
            let givens_mat = Matrix::new(2, 2, vec![c, -s, s, c]);

            {
                // Apply Givens rotation to the block (on the left)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [q, p - 2], 2, n - p + 2);
                let transformed = &givens_mat * &h_block;
                h_block.set_to(transformed.as_slice());
            }

            {
                // Apply Givens rotation to block (on the right)
                let h_block = MatrixSliceMut::from_matrix(&mut h, [0, q], p + 1, 2);
                let transformed = &h_block * givens_mat.transpose();
                h_block.set_to(transformed.as_slice());
            }

            {
                // Update the transformation matrix
                let trans_block = MatrixSliceMut::from_matrix(&mut transformation, [0, q], n, 2);
                let transformed = &trans_block * givens_mat.transpose();
                trans_block.set_to(transformed.as_slice());
            }

            // Check for convergence
            if abs(h[[p, q]]) < eps * (abs(h[[q, q]]) + abs(h[[p, p]])) {
                h.data[p * h.cols + q] = T::zero();
                p -= 1;
            } else if abs(h[[p - 1, q - 1]]) < eps * (abs(h[[q - 1, q - 1]]) + abs(h[[q, q]])) {
                h.data[(p - 1) * h.cols + q - 1] = T::zero();
                p -= 2;
            }
        }

        Ok((h.diag().into_vec(), u * transformation))
    }

    /// Eigendecomposition of a square matrix.
    ///
    /// Returns a Vec of eigenvalues, and a matrix with eigenvectors as the columns.
    ///
    /// The eigenvectors are only gauranteed to be correct if the matrix is real-symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::matrix::Matrix;
    ///
    /// let a = Matrix::new(3,3,vec![3.,2.,4.,2.,0.,2.,4.,2.,3.]);
    ///
    /// let (e, m) = a.eigendecomp().expect("We should be able to compute this eigendecomp!");
    /// println!("{:?}", e);
    /// println!("{:?}", m.data());
    /// ```
    ///
    /// # Panics
    ///
    /// - The matrix is not square.
    ///
    /// # Failures
    ///
    /// - The eigen decomposition can not be computed.
    pub fn eigendecomp(&self) -> Result<(Vec<T>, Matrix<T>), Error> {
        let n = self.rows();
        assert!(n == self.cols, "Matrix must be square for eigendecomp.");

        match n {
            1 => Ok((vec![self.data[0]], Matrix::new(1, 1, vec![T::one()]))),
            2 => self.direct_2_by_2_eigendecomp(),
            _ => self.francis_shift_eigendecomp(),
        }
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
/// use rulinalg::matrix::Matrix;
///
/// let a = Matrix::new(3,3, vec![1.0,2.0,0.0,
///                               0.0,3.0,4.0,
///                               5.0, 1.0, 2.0]);
///
/// let (l,u,p) = a.lup_decomp().expect("This matrix should decompose!");
/// ```
///
/// # Panics
///
/// - Matrix is not square.
///
/// # Failures
///
/// - Matrix cannot be LUP decomposed.
    pub fn lup_decomp(&self) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), Error> {
        let n = self.cols;
        assert!(self.rows == n, "Matrix must be square for LUP decomposition.");

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
                    return Err(Error::new(ErrorKind::DecompFailure,
                        "Matrix could not be LUP decomposed."));
                }
                l.data[j*n + i] = (a_2[[j,i]] - s2) / denom;
            }

        }

        Ok((l,u,p))
    }
}



#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use vector::Vector;

    #[test]
    fn test_1_by_1_matrix_eigenvalues() {
        let a = Matrix::new(1, 1, vec![3.]);
        assert_eq!(vec![3.], a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_eigenvalues() {
        let a = Matrix::new(2, 2, vec![1., 2., 3., 4.]);
        // characteristic polynomial is λ² − 5λ − 2 = 0
        assert_eq!(vec![(5. - (33.0f32).sqrt()) / 2., (5. + (33.0f32).sqrt()) / 2.],
                   a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_zeros_eigenvalues() {
        let a = Matrix::new(2, 2, vec![0.; 4]);
        // characteristic polynomial is λ² = 0
        assert_eq!(vec![0.0, 0.0], a.eigenvalues().unwrap());
    }

    #[test]
    fn test_2_by_2_matrix_complex_eigenvalues() {
        // This test currently fails - complex eigenvalues would be nice though!
        let a = Matrix::new(2, 2, vec![1.0, -3.0, 1.0, 1.0]);
        // characteristic polynomial is λ² − λ + 4 = 0

        // Decomposition will fail
        assert!(a.eigenvalues().is_err());
    }

    #[test]
    fn test_2_by_2_matrix_eigendecomp() {
        let a = Matrix::new(2, 2, vec![20., 4., 20., 16.]);
        let (eigenvals, eigenvecs) = a.eigendecomp().unwrap();

        let lambda_1 = eigenvals[0];
        let lambda_2 = eigenvals[1];

        let v1 = Vector::new(vec![eigenvecs[[0, 0]], eigenvecs[[1, 0]]]);
        let v2 = Vector::new(vec![eigenvecs[[0, 1]], eigenvecs[[1, 1]]]);

        let epsilon = 0.00001;
        assert!((&a * &v1 - &v1 * lambda_1).into_vec().iter().all(|&c| c < epsilon));
        assert!((&a * &v2 - &v2 * lambda_2).into_vec().iter().all(|&c| c < epsilon));
    }

    #[test]
    fn test_3_by_3_eigenvals() {
        let a = Matrix::new(3, 3, vec![17f64, 22., 27., 22., 29., 36., 27., 36., 45.]);

        let eigs = a.eigenvalues().unwrap();

        let eig_1 = 90.4026;
        let eig_2 = 0.5973;
        let eig_3 = 0.0;

        assert!(eigs.iter().any(|x| (x - eig_1).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_2).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_3).abs() < 1e-4));
    }

    #[test]
    fn test_5_by_5_eigenvals() {
        let a = Matrix::new(5,
                            5,
                            vec![1f64, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 1.0,
                                 7.0, 1.0, 1.0, 4.0, 2.0, 1.0, -1.0, 3.0, 5.0, 1.0, 1.0, 3.0, 2.0]);

        let eigs = a.eigenvalues().unwrap();

        let eig_1 = 12.174;
        let eig_2 = 5.2681;
        let eig_3 = -4.4942;
        let eig_4 = 2.9279;
        let eig_5 = -2.8758;

        assert!(eigs.iter().any(|x| (x - eig_1).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_2).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_3).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_4).abs() < 1e-4));
        assert!(eigs.iter().any(|x| (x - eig_5).abs() < 1e-4));
    }

    #[test]
    #[should_panic]
    fn test_non_square_cholesky() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.cholesky();
    }

    #[test]
    #[should_panic]
    fn test_non_square_upper_hessenberg() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.upper_hessenberg();
    }

    #[test]
    #[should_panic]
    fn test_non_square_upper_hess_decomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.upper_hess_decomp();
    }

    #[test]
    #[should_panic]
    fn test_non_square_eigenvalues() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.eigenvalues();
    }

    #[test]
    #[should_panic]
    fn test_non_square_eigendecomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.eigendecomp();
    }

    #[test]
    #[should_panic]
    fn test_non_square_lup_decomp() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);

        let _ = a.lup_decomp();
    }
}
