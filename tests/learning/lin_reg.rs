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

    lin_mod.train(&inputs, &targets).unwrap();

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

    let _ = lin_mod.predict(&inputs).unwrap();
}

#[cfg(feature = "datasets")]
#[test]
fn test_regression_datasets_trees() {
    use rm::datasets::trees;
    let trees = trees::load();

    let mut lin_mod = LinRegressor::default();
    lin_mod.train(&trees.data(), &trees.target()).unwrap();
    let params = lin_mod.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-57.98765891838409, 4.708160503017506, 0.3392512342447438]));

    let predicted = lin_mod.predict(&trees.data()).unwrap();
    let expected = vec![4.837659653793278, 4.55385163347481, 4.816981265588826, 15.874115228921276,
                        19.869008437727473, 21.018326956518717, 16.192688074961563, 19.245949183164257,
                        21.413021404689726, 20.187581283767756, 22.015402271048487, 21.468464618616007,
                        21.468464618616007, 20.50615412980805, 23.954109686181766, 27.852202904652785,
                        31.583966481344966, 33.806481916796706, 30.60097760433255, 28.697035014921106,
                        34.388184394951004, 36.008318964043994, 35.38525970948079, 41.76899799551756,
                        44.87770231764652, 50.942867757643015, 52.223751092491256, 53.42851282520877,
                        53.899328875510534, 53.899328875510534, 68.51530482306926];
    assert_eq!(predicted, Vector::new(expected));
}

#[test]
#[ignore = "FIXME #183 fails nondeterministically"]
fn test_train_no_data() {
    let inputs = Matrix::new(0, 1, vec![]);
    let targets = Vector::new(vec![]);

    let mut lin_mod = LinRegressor::default();
    let res = lin_mod.train(&inputs, &targets);

    assert!(res.is_err());
}
