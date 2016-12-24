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

#[test]
fn test_regression_trees_data() {
    // test using trees datasets
    // Atkinson, A. C. (1985) Plots, Transformations and Regression. Oxford University Press.

    // girth, height
    let data = matrix![8.3, 70.;
                       8.6, 65.;
                       8.8, 63.;
                       10.5, 72.;
                       10.7, 81.;
                       10.8, 83.;
                       11.0, 66.;
                       11.0, 75.;
                       11.1, 80.;
                       11.2, 75.;
                       11.3, 79.;
                       11.4, 76.;
                       11.4, 76.;
                       11.7, 69.;
                       12.0, 75.;
                       12.9, 74.;
                       12.9, 85.;
                       13.3, 86.;
                       13.7, 71.;
                       13.8, 64.;
                       14.0, 78.;
                       14.2, 80.;
                       14.5, 74.;
                       16.0, 72.;
                       16.3, 77.;
                       17.3, 81.;
                       17.5, 82.;
                       17.9, 80.;
                       18.0, 80.;
                       18.0, 80.;
                       20.6, 87.];
    let volume = vec![10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9,
                      24.2, 21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9,
                      34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0,
                      77.0];
    let mut lin_mod = LinRegressor::default();
    lin_mod.train(&data, &Vector::new(volume)).unwrap();
    let params = lin_mod.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-57.98765891838409, 4.708160503017506, 0.3392512342447438]));

    let predicted = lin_mod.predict(&data).unwrap();
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