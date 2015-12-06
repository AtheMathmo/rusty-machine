extern crate rusty_collaborative_filter as rcf;

use rcf::math::linalg::matrix::Matrix;
use rcf::math::linalg::vector::Vector;

#[test]
fn create_mat_new() {
	let a = Matrix::new(4, 3, vec![0.0; 12]);

	assert_eq!(a.rows, 4);
	assert_eq!(a.cols, 3);
}

#[test]
fn create_mat_zeros() {
	let a = Matrix::zeros(10, 10);

	assert_eq!(a.rows, 10);
	assert_eq!(a.cols, 10);

	for i in 0..10
	{
		for j in 0..10
		{
			assert_eq!(a[[i,j]], 0.0);
		}
	}
}

#[test]
fn create_mat_identity() {
	let a = Matrix::identity(4);

	assert_eq!(a.rows, 4);
	assert_eq!(a.cols, 4);

	assert_eq!(a[[0,0]], 1.0);
	assert_eq!(a[[1,1]], 1.0);
	assert_eq!(a[[2,2]], 1.0);
	assert_eq!(a[[3,3]], 1.0);

	assert_eq!(a[[0,1]], 0.0);
	assert_eq!(a[[2,1]], 0.0);
	assert_eq!(a[[3,0]], 0.0);
}

#[test]
fn create_mat_diag() {
	let a = Matrix::from_diag(&[1.0,2.0,3.0,4.0]);

	assert_eq!(a.rows, 4);
	assert_eq!(a.cols, 4);

	assert_eq!(a[[0,0]], 1.0);
	assert_eq!(a[[1,1]], 2.0);
	assert_eq!(a[[2,2]], 3.0);
	assert_eq!(a[[3,3]], 4.0);

	assert_eq!(a[[0,1]], 0.0);
	assert_eq!(a[[2,1]], 0.0);
	assert_eq!(a[[3,0]], 0.0);
}

#[test]
fn transpose_mat() {
	let a = Matrix::new(5, 2, vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]);

	let c = a.transpose();

	assert_eq!(c.cols, a.rows);
	assert_eq!(c.rows, a.cols);

	assert_eq!(a[[0,0]], c[[0,0]]);
	assert_eq!(a[[1,0]], c[[0,1]]);
	assert_eq!(a[[2,0]], c[[0,2]]);
	assert_eq!(a[[3,0]], c[[0,3]]);
	assert_eq!(a[[4,0]], c[[0,4]]);
	assert_eq!(a[[0,1]], c[[1,0]]);
	assert_eq!(a[[1,1]], c[[1,1]]);
	assert_eq!(a[[2,1]], c[[1,2]]);
	assert_eq!(a[[3,1]], c[[1,3]]);
	assert_eq!(a[[4,1]], c[[1,4]]);

}

#[test]
fn indexing_mat() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);

	assert_eq!(a[[0,0]], 1.0);
	assert_eq!(a[[0,1]], 2.0);
	assert_eq!(a[[1,0]], 3.0);
	assert_eq!(a[[1,1]], 4.0);
	assert_eq!(a[[2,0]], 5.0);
	assert_eq!(a[[2,1]], 6.0);
}

#[test]
fn matrix_mul() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = Matrix::new(2, 3, vec![1.,2.,3.,4.,5.,6.]);

	let c = a * b;

	assert_eq!(c.rows, 3);
	assert_eq!(c.cols, 3);

	assert_eq!(c[[0,0]], 9.0);
	assert_eq!(c[[0,1]], 12.0);
	assert_eq!(c[[0,2]], 15.0);
	assert_eq!(c[[1,0]], 19.0);
	assert_eq!(c[[1,1]], 26.0);
	assert_eq!(c[[1,2]], 33.0);
	assert_eq!(c[[2,0]], 29.0);
	assert_eq!(c[[2,1]], 40.0);
	assert_eq!(c[[2,2]], 51.0);
}

#[test]
fn matrix_vec_mul() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = Vector::new(vec![4.,7.]);

	let c = a * b;

	assert_eq!(c.size, 3);

	assert_eq!(c[0], 18.0);
	assert_eq!(c[1], 40.0);
	assert_eq!(c[2], 62.0);
}

#[test]
fn matrix_f32_mul() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);

	let c = a * 2.0;

	assert_eq!(c[[0,0]], 2.0);
	assert_eq!(c[[0,1]], 4.0);
	assert_eq!(c[[1,0]], 6.0);
	assert_eq!(c[[1,1]], 8.0);
	assert_eq!(c[[2,0]], 10.0);
	assert_eq!(c[[2,1]], 12.0);
}

#[test]
fn matrix_add() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = Matrix::new(3, 2, vec![2.,3.,4.,5.,6.,7.]);

	let c = a + b;

	assert_eq!(c[[0,0]], 3.0);
	assert_eq!(c[[0,1]], 5.0);
	assert_eq!(c[[1,0]], 7.0);
	assert_eq!(c[[1,1]], 9.0);
	assert_eq!(c[[2,0]], 11.0);
	assert_eq!(c[[2,1]], 13.0);
}

#[test]
fn matrix_f32_add() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = 3.0;

	let c = a + b;

	assert_eq!(c[[0,0]], 4.0);
	assert_eq!(c[[0,1]], 5.0);
	assert_eq!(c[[1,0]], 6.0);
	assert_eq!(c[[1,1]], 7.0);
	assert_eq!(c[[2,0]], 8.0);
	assert_eq!(c[[2,1]], 9.0);
}

#[test]
fn matrix_sub() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = Matrix::new(3, 2, vec![2.,3.,4.,5.,6.,7.]);

	let c = a - b;

	assert_eq!(c[[0,0]], -1.0);
	assert_eq!(c[[0,1]], -1.0);
	assert_eq!(c[[1,0]], -1.0);
	assert_eq!(c[[1,1]], -1.0);
	assert_eq!(c[[2,0]], -1.0);
	assert_eq!(c[[2,1]], -1.0);
}

#[test]
fn matrix_f32_sub() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = 3.0;

	let c = a - b;

	assert_eq!(c[[0,0]], -2.0);
	assert_eq!(c[[0,1]], -1.0);
	assert_eq!(c[[1,0]], 0.0);
	assert_eq!(c[[1,1]], 1.0);
	assert_eq!(c[[2,0]], 2.0);
	assert_eq!(c[[2,1]], 3.0);
}

#[test]
fn matrix_f32_div() {
	let a = Matrix::new(3, 2, vec![1.,2.,3.,4.,5.,6.]);
	let b = 3.0;

	let c = a / b;

	assert_eq!(c[[0,0]], 1.0/3.0);
	assert_eq!(c[[0,1]], 2.0/3.0);
	assert_eq!(c[[1,0]], 1.0);
	assert_eq!(c[[1,1]], 4.0/3.0);
	assert_eq!(c[[2,0]], 5.0/3.0);
	assert_eq!(c[[2,1]], 2.0);
}