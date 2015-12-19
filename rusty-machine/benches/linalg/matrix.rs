use rm::linalg::matrix::Matrix;
use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

#[bench]
fn mat_mul(b: &mut Bencher) {

	let a = Matrix::new(10,10, vec![2.0;100]);
	let c = Matrix::new(10,10, vec![3.0; 100]);

    b.iter(|| &a * &c)
}