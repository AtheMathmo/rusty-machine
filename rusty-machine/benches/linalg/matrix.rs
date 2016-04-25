use rm::linalg::matrix::Matrix;
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

macro_rules! mat_mul (
    ($fn_name:ident, $para_fn_name:ident, $n:expr, $k:expr, $m:expr) => (
#[bench]
fn $fn_name(b: &mut Bencher) {
    let a = Matrix::new($n, $k, vec![2f32; $n * $k]);
    let c = Matrix::new($k, $m, vec![3f32; $k * $m]);

    b.iter(|| &a * &c)
}

#[bench]
fn $para_fn_name(b: &mut Bencher) {
    let a = Matrix::new($n, $k, vec![2f32; $n * $k]);
    let c = Matrix::new($k, $m, vec![3f32; $k * $m]);

    b.iter(|| a.paramul(&c))
}
    );
);

mat_mul!(mat_mul_10_10_10, mat_paramul_10_10_10, 10,10,10);
mat_mul!(mat_mul_128_100_128, mat_paramul_128_100_128, 128,100,128);
mat_mul!(mat_mul_128_1000_128, mat_paramul_128_1000_128, 128,1000,128);
mat_mul!(mat_mul_128_10000_128, mat_paramul_128_10000_128, 128,10000,128);
mat_mul!(mat_mul_128_100000_128, mat_paramul_128_100000_128, 128,100000,128);
mat_mul!(mat_mul_5000_20_20, mat_paramul_5000_20_20, 5000,20,20);
mat_mul!(mat_mul_20_20_5000, mat_paramul_20_20_5000, 20,20,5000);
