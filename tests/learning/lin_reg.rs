use rm::linalg::Matrix;
use rm::linalg::Vector;
use rm::learning::SupModel;
use rm::learning::lin_reg::LinRegressor;
use libnum::abs;

#[test]
fn test_optimized_regression() {
    let mut lin_mod = LinRegressor::default();
    let inputs = Matrix::new(3, 1, vec![2.0, 3.0, 4.0]);
    let targets = Vector::new(vec![5.0, 6.0, 7.0]);

    lin_mod.train_with_optimization(&inputs, &targets);

    let _ = lin_mod.parameters().unwrap();
}

#[test]
fn test_regression() {
    let mut lin_mod = LinRegressor::default();
    let inputs = Matrix::new(3, 1, vec![2.0, 3.0, 4.0]);
    let targets = Vector::new(vec![5.0, 6.0, 7.0]);

    lin_mod.train(&inputs, &targets);

    let parameters = lin_mod.parameters().unwrap();

    let err_1 = abs(parameters[0] - 3.0);
    let err_2 = abs(parameters[1] - 1.0);

    assert!(err_1 < 1e-8);
    assert!(err_2 < 1e-8);
}

#[test]
#[should_panic]
fn test_no_train_params() {
    let lin_mod = LinRegressor::default();

    let _ = lin_mod.parameters().unwrap();
}

#[test]
#[should_panic]
fn test_no_train_predict() {
    let lin_mod = LinRegressor::default();
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    let _ = lin_mod.predict(&inputs);
}