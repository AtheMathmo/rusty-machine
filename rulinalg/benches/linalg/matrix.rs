use rulinalg::matrix::Matrix;
use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

#[bench]
fn mat_ref_add_100_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	&a + &c
    })
}

#[bench]
fn mat_create_add_100_100(b: &mut Bencher) {
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a + &c
    })
}

#[bench]
fn mat_create_100_100(b: &mut Bencher) {
    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a
    })
}

#[bench]
fn mat_mul_10_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2f32; 100]);
    let c = Matrix::new(10, 10, vec![3f32; 100]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_mul_128_100(b: &mut Bencher) {

    let a = Matrix::new(128, 100, vec![2f32; 12800]);
    let c = Matrix::new(100, 128, vec![3f32; 12800]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_mul_128_1000(b: &mut Bencher) {

    let a = Matrix::new(128, 1000, vec![2f32; 128000]);
    let c = Matrix::new(1000, 128, vec![3f32; 128000]);

    b.iter(|| &a * &c)
}