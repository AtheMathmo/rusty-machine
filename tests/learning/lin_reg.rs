use rm::linalg::Matrix;
use rm::linalg::Vector;
use rm::learning::SupModel;
use rm::learning::lin_reg::{LinRegressor, RidgeRegressor};
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
fn test_linear_regression_datasets_trees() {
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


#[cfg(feature = "datasets")]
#[test]
fn test_ridge_regression_datasets_trees() {
    use rm::datasets::trees;
    let trees = trees::load();

    let mut lin_mod = RidgeRegressor::default();
    lin_mod.train(&trees.data(), &trees.target()).unwrap();
    let params = lin_mod.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-58.09806161950894, 4.68684745409343, 0.34441921086952676]));

    let predicted = lin_mod.predict(&trees.data()).unwrap();
    let expected = vec![4.9121170103334, 4.596075192213792, 4.844606261293432, 15.912019831077998, 19.949162219722425,
                        21.106685386870826, 16.18892829290755, 19.288701190733292, 21.479481990490267, 20.226070681551974,
                        22.072432270439435, 21.5078593832402, 21.5078593832402, 20.502979143381534, 23.975548644826723,
                        27.84929214264129, 31.637903462206083, 33.85706165471298, 30.565512473307443, 28.623262742630104,
                        34.38250118562216, 36.00870909817989, 35.34824806919077, 41.68968082859186, 44.81783111916752,
                        50.882355416739074, 52.16414411842728, 53.35004467832559, 53.818729423734936, 53.818729423734936,
                        68.41546728046455];
    assert_eq!(predicted, Vector::new(expected));
}

#[cfg(feature = "datasets")]
#[test]
fn test_ridge_regression_datasets_trees_alpha01() {
    use rm::datasets::trees;
    let trees = trees::load();

    let mut lin_mod = RidgeRegressor::new(0.1);
    lin_mod.train(&trees.data(), &trees.target()).unwrap();
    let params = lin_mod.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-57.99878658933356, 4.706019761728981, 0.3397708268791373]));

    let predicted = lin_mod.predict(&trees.data()).unwrap();
    let expected = vec![4.84513531455659, 4.558087108679594, 4.819749407267118, 15.877920444118622, 19.877061838376648,
                        21.027205468307827, 16.19230536370829, 19.250242805620523, 21.419698916189105, 20.19144675796631,
                        22.02113204165577, 21.472421537191252, 21.472421537191252, 20.50583167755598, 23.956262567349498,
                        27.85190952602645, 31.589388621696962, 33.81156735326769, 30.597412854772223, 28.689619042791165,
                        34.38761457144487, 36.00836017754894, 35.38154114479282, 41.761029133628014, 44.871689196542405,
                        50.936792265787936, 52.21776704501286, 53.420633295946175, 53.891235272119076, 53.891235272119076,
                        68.50528244076838];
    assert_eq!(predicted, Vector::new(expected));
}

#[cfg(feature = "datasets")]
#[test]
fn test_ridge_regression_datasets_trees_alpha00() {
    // should be the same as LinRegressor
    use rm::datasets::trees;
    let trees = trees::load();

    let mut lin_mod = RidgeRegressor::new(0.0);
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