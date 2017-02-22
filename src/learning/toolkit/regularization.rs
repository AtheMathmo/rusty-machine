//! Regularization Module
//!
//! This module contains some base utility methods for regularization
//! within machine learning algorithms.
//!
//! The module contains a `Regularization` enum which provides access to
//! `L1`, `L2` and `ElasticNet` regularization.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::toolkit::regularization::Regularization;
//!
//! let reg = Regularization::L1(0.5);
//! ```

use linalg::norm::{Euclidean, Lp, MatrixNorm};
use linalg::{Matrix, MatrixSlice, BaseMatrix};
use libnum::{FromPrimitive, Float};

/// Model Regularization
#[derive(Debug, Clone, Copy)]
pub enum Regularization<T: Float> {
    /// L1 Regularization
    L1(T),
    /// L2 Regularization
    L2(T),
    /// Elastic Net Regularization (L1 and L2)
    ElasticNet(T, T),
    /// No Regularization
    None,
}

impl<T: Float + FromPrimitive> Regularization<T> {
    /// Compute the regularization addition to the cost.
    pub fn reg_cost(&self, mat: MatrixSlice<T>) -> T {
        match *self {
            Regularization::L1(x) => Self::l1_reg_cost(&mat, x),
            Regularization::L2(x) => Self::l2_reg_cost(&mat, x),
            Regularization::ElasticNet(x, y) => {
                Self::l1_reg_cost(&mat, x) + Self::l2_reg_cost(&mat, y)
            }
            Regularization::None => T::zero(),
        }
    }

    /// Compute the regularization addition to the gradient.
    pub fn reg_grad(&self, mat: MatrixSlice<T>) -> Matrix<T> {
        match *self {
            Regularization::L1(x) => Self::l1_reg_grad(&mat, x),
            Regularization::L2(x) => Self::l2_reg_grad(&mat, x),
            Regularization::ElasticNet(x, y) => {
                Self::l1_reg_grad(&mat, x) + Self::l2_reg_grad(&mat, y)
            }
            Regularization::None => Matrix::zeros(mat.rows(), mat.cols()),
        }
    }

    fn l1_reg_cost(mat: &MatrixSlice<T>, x: T) -> T {
        let l1_norm = Lp::Integer(1).norm(mat);
        l1_norm * x / ((T::one() + T::one()) * FromPrimitive::from_usize(mat.rows()).unwrap())
    }

    fn l1_reg_grad(mat: &MatrixSlice<T>, x: T) -> Matrix<T> {
        let m_2 = (T::one() + T::one()) * FromPrimitive::from_usize(mat.rows()).unwrap();
        let out_mat_data = mat.iter()
            .map(|y| {
                if y.is_sign_negative() {
                    -x / m_2
                } else {
                    x / m_2
                }
            })
            .collect::<Vec<_>>();
        Matrix::new(mat.rows(), mat.cols(), out_mat_data)
    }

    fn l2_reg_cost(mat: &MatrixSlice<T>, x: T) -> T {
        Euclidean.norm(mat) * x / ((T::one() + T::one()) * FromPrimitive::from_usize(mat.rows()).unwrap())
    }

    fn l2_reg_grad(mat: &MatrixSlice<T>, x: T) -> Matrix<T> {
        mat * (x / FromPrimitive::from_usize(mat.rows()).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::Regularization;
    use linalg::{Matrix, BaseMatrix};
    use linalg::norm::{Euclidean, MatrixNorm};

    #[test]
    fn test_no_reg() {
        let input_mat = Matrix::new(3, 4, (0..12).map(|x| x as f64).collect::<Vec<_>>());
        let mat_slice = input_mat.as_slice();

        let no_reg: Regularization<f64> = Regularization::None;

        let a = no_reg.reg_cost(mat_slice);
        let b = no_reg.reg_grad(mat_slice);

        assert_eq!(a, 0f64);
        assert_eq!(b, Matrix::zeros(3, 4));
    }

    #[test]
    fn test_l1_reg() {
        let input_mat = Matrix::new(3, 4, (0..12).map(|x| x as f64 - 3f64).collect::<Vec<_>>());
        let mat_slice = input_mat.as_slice();

        let no_reg: Regularization<f64> = Regularization::L1(0.5);

        let a = no_reg.reg_cost(mat_slice);
        let b = no_reg.reg_grad(mat_slice);

        assert!((a - (42f64 / 12f64)) < 1e-18);

        let true_grad = vec![-1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            .into_iter()
            .map(|x| x / 12f64)
            .collect::<Vec<_>>();

        for eps in (b - Matrix::new(3, 4, true_grad)).into_vec() {
            assert!(eps < 1e-18);
        }
    }

    #[test]
    fn test_l2_reg() {
        let input_mat = Matrix::new(3, 4, (0..12).map(|x| x as f64 - 3f64).collect::<Vec<_>>());
        let mat_slice = input_mat.as_slice();

        let no_reg: Regularization<f64> = Regularization::L2(0.5);

        let a = no_reg.reg_cost(mat_slice);
        let b = no_reg.reg_grad(mat_slice);

        assert!((a - (Euclidean.norm(&input_mat) / 12f64)) < 1e-18);

        let true_grad = &input_mat / 6f64;
        for eps in (b - true_grad).into_vec() {
            assert!(eps < 1e-18);
        }
    }

    #[test]
    fn test_elastic_net_reg() {
        let input_mat = Matrix::new(3, 4, (0..12).map(|x| x as f64 - 3f64).collect::<Vec<_>>());
        let mat_slice = input_mat.as_slice();

        let no_reg: Regularization<f64> = Regularization::ElasticNet(0.5, 0.25);

        let a = no_reg.reg_cost(mat_slice);
        let b = no_reg.reg_grad(mat_slice);

        assert!(a - ((Euclidean.norm(&input_mat) / 24f64) + (42f64 / 12f64)) < 1e-18);

        let l1_true_grad = Matrix::new(3, 4,
            vec![-1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            .into_iter()
            .map(|x| x / 12f64)
            .collect::<Vec<_>>());
        let l2_true_grad = &input_mat / 12f64;

        for eps in (b - l1_true_grad - l2_true_grad)
            .into_vec() {
            // Slightly lower boundary than others - more numerical error as more ops.
            assert!(eps < 1e-12);
        }
    }
}
