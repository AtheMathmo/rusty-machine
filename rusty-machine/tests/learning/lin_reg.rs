use rm::linalg::matrix::Matrix;
use rm::linalg::vector::Vector;
use rm::learning::SupModel;
use rm::learning::lin_reg::LinRegressor;
use libnum::abs;

#[test]
fn test_regression() {
    let mut lin_mod = LinRegressor::new();
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let values = Vector::new(vec![5.0, 6.0, 7.0]);

    lin_mod.train(&data, &values);

    let b_val = lin_mod.b.unwrap();

    let err_1 = abs(b_val[0] - 3.0);
    let err_2 = abs(b_val[1] - 1.0);

    assert!(err_1 < 1e-8);
    assert!(err_2 < 1e-8);
}
