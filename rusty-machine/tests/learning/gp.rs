use rm::linalg::matrix::Matrix;
use rm::learning::SupModel;
use rm::learning::gp::GaussianProcess;

#[test]
fn test_default_gp() {
	let mut gp = GaussianProcess::default();

	let train_data = Matrix::new(10,1,vec![0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]);
	let target = Matrix::new(10,1,vec![0.,1.,2.,3.,4.,4.,3.,2.,1.,0.]);

	gp.train(&train_data, &target);

	let test_data = Matrix::new(5,1,vec![2.3,4.4,5.1,6.2,7.1]);

	let output = gp.predict(&test_data);
}