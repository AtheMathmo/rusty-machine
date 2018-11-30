use rm::learning::pca::PCA;
use rm::learning::UnSupModel;
use rm::linalg::Matrix;

#[test]
fn test_default() {
    let mut model = PCA::default();

    let inputs = Matrix::new(
        7,
        3,
        vec![
            8.3, 50., 23., 10.2, 55., 21., 11.1, 57., 22., 12.5, 60., 15., 11.3, 59., 20., 12.4,
            61., 11., 11.2, 58., 23.,
        ],
    );
    model.train(&inputs).unwrap();

    let cexp = Matrix::new(
        3,
        3,
        vec![
            0.2304196717022202,
            0.2504639278931734,
            -0.9403055863478447,
            0.5897383434061588,
            0.7326863014098074,
            0.3396755364211204,
            -0.7740254913174374,
            0.6328021843757651,
            -0.021117155112842168,
        ],
    );
    let cmp = model.components().unwrap();
    assert_matrix_eq!(cmp, cexp, comp = abs, tol = 1e-8);

    let new_data = Matrix::new(1, 3, vec![9., 45., 22.]);
    let outputs = model.predict(&new_data).unwrap();

    let exp = Matrix::new(
        1,
        3,
        vec![-9.72287413262656, -7.680227015314077, -2.301338333438487],
    );
    assert_matrix_eq!(outputs, exp, comp = abs, tol = 1e-8);
}

#[test]
fn test_not_centering() {
    let mut model = PCA::new(3, false);

    let inputs = Matrix::new(
        7,
        3,
        vec![
            8.3, 50., 23., 10.2, 55., 21., 11.1, 57., 22., 12.5, 60., 15., 11.3, 59., 20., 12.4,
            61., 11., 11.2, 58., 23.,
        ],
    );
    model.train(&inputs).unwrap();

    let cexp = Matrix::new(
        3,
        3,
        vec![
            0.17994480617740657,
            -0.16908609066166264,
            0.9690354795746806,
            0.9326216647416523,
            -0.2839205184846983,
            -0.2227239763426676,
            0.3127885822473139,
            0.9438215049087068,
            0.10660332868901998,
        ],
    );
    let cmp = model.components().unwrap();
    assert_matrix_eq!(cmp, cexp, comp = abs, tol = 1e-8);

    let new_data = Matrix::new(1, 3, vec![9., 45., 22.]);
    let outputs = model.predict(&new_data).unwrap();

    let exp = Matrix::new(
        1,
        3,
        vec![50.468826978411926, 6.465874960225161, 1.0440136119105228],
    );
    assert_matrix_eq!(outputs, exp, comp = abs, tol = 1e-8);
}

#[test]
fn test_filter_component() {
    let mut model = PCA::new(2, false);

    let inputs = Matrix::new(
        7,
        3,
        vec![
            8.3, 50., 23., 10.2, 55., 21., 11.1, 57., 22., 12.5, 60., 15., 11.3, 59., 20., 12.4,
            61., 11., 11.2, 58., 23.,
        ],
    );
    model.train(&inputs).unwrap();

    let cexp = Matrix::new(
        3,
        2,
        vec![
            0.17994480617740657,
            -0.16908609066166264,
            0.9326216647416523,
            -0.2839205184846983,
            0.3127885822473139,
            0.9438215049087068,
        ],
    );
    let cmp = model.components().unwrap();
    assert_matrix_eq!(cmp, cexp, comp = abs, tol = 1e-8);

    let new_data = Matrix::new(1, 3, vec![9., 45., 22.]);
    let outputs = model.predict(&new_data).unwrap();

    let exp = Matrix::new(1, 2, vec![50.468826978411926, 6.465874960225161]);
    assert_matrix_eq!(outputs, exp, comp = abs, tol = 1e-8);
}

#[test]
fn test_predict_different_dimension() {
    let mut model = PCA::new(2, false);

    let inputs = Matrix::new(
        7,
        3,
        vec![
            8.3, 50., 23., 10.2, 55., 21., 11.1, 57., 22., 12.5, 60., 15., 11.3, 59., 20., 12.4,
            61., 11., 11.2, 58., 23.,
        ],
    );
    model.train(&inputs).unwrap();

    let new_data = Matrix::new(1, 2, vec![1., 2.]);
    let err = model.predict(&new_data);
    assert!(err.is_err());

    let new_data = Matrix::new(1, 4, vec![1., 2., 3., 4.]);
    let err = model.predict(&new_data);
    assert!(err.is_err());

    let mut model = PCA::new(5, false);
    let err = model.train(&inputs);
    assert!(err.is_err());
}

#[test]
fn test_wide() {
    let mut model = PCA::default();

    let inputs = Matrix::new(2, 4, vec![8.3, 50., 23., 2., 10.2, 55., 21., 3.]);
    model.train(&inputs).unwrap();

    let cexp = Matrix::new(
        2,
        4,
        vec![
            0.3277323746171723,
            0.8624536174136117,
            -0.3449814469654447,
            0.17249072348272235,
            0.933710591152088,
            -0.23345540994181946,
            0.23959824886246414,
            -0.1275765757549414,
        ],
    );
    let cmp = model.components().unwrap();
    assert_matrix_eq!(cmp, cexp, comp = abs, tol = 1e-8);

    let new_data = Matrix::new(1, 4, vec![9., 45., 22., 2.5]);
    let outputs = model.predict(&new_data).unwrap();

    let exp = Matrix::new(1, 2, vec![-6.550335224256381, 1.517487926775624]);
    assert_matrix_eq!(outputs, exp, comp = abs, tol = 1e-8);
}
