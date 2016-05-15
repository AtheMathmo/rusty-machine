use std::convert::From;

use super::matrix::{Matrix, MatrixSlice, MatrixSliceMut};
use super::vector::Vector;


impl<T> From<Vector<T>> for Matrix<T> {
    fn from(vector: Vector<T>) -> Self {
        Matrix::new(vector.size(), 1, vector.into_vec())
    }
}


#[cfg(test)]
mod tests {
    use super::super::matrix::Matrix;
    use super::super::vector::Vector;

    #[test]
    fn inner_product_as_matrix_multiplication() {
        let u: Vector<f32> = Vector::new(vec![1., 2., 3.]);
        let v: Vector<f32> = Vector::new(vec![3., 4., 5.]);
        let dot_product = u.dot(&v);

        let um: Matrix<f32> = u.into();
        let vm: Matrix<f32> = v.into();
        let matrix_product = um.transpose() * vm;

        assert_eq!(dot_product, matrix_product.data()[0]);
    }

}
