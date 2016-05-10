use linalg::Metric;
use linalg::matrix::{Matrix, MatrixSlice};
use linalg::matrix::slice::BaseSlice;
use linalg::utils;
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
        match self {
            &Regularization::L1(x) => Self::l1_reg_cost(&mat, x),
            &Regularization::L2(x) => Self::l2_reg_cost(&mat, x),
            &Regularization::ElasticNet(x, y) => {
                Self::l1_reg_cost(&mat, x) + Self::l2_reg_cost(&mat, y)
            }
            &Regularization::None => T::zero(),
        }
    }

    /// Compute the regularization addition to the gradient.
    pub fn reg_grad(&self, mat: MatrixSlice<T>) -> Matrix<T> {
        match self {
            &Regularization::L1(x) => Self::l1_reg_grad(&mat, x),
            &Regularization::L2(x) => Self::l2_reg_grad(&mat, x),
            &Regularization::ElasticNet(x, y) => {
                Self::l1_reg_grad(&mat, x) + Self::l2_reg_grad(&mat, y)
            }
            &Regularization::None => Matrix::zeros(mat.rows(), mat.cols()),
        }
    }

    fn l1_reg_cost(mat: &MatrixSlice<T>, x: T) -> T {
        let l1_norm = mat.iter_rows()
                         .fold(T::zero(), |acc, row| acc + utils::unrolled_sum(row));
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
        mat.norm() * x / ((T::one() + T::one()) * FromPrimitive::from_usize(mat.rows()).unwrap())
    }

    fn l2_reg_grad(mat: &MatrixSlice<T>, x: T) -> Matrix<T> {
        mat * (x / FromPrimitive::from_usize(mat.rows()).unwrap())
    }
}
