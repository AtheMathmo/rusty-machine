use rm::linalg::matrix::Matrix;
use rm::learning::UnSupModel;
use rm::learning::k_means::KMeansClassifier;
use rm::learning::k_means::InitAlgorithm;

#[test]
fn test_model_default() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&data);

    let a = model.predict(&pred_data);

    assert_eq!(a.data.len(), 3);
}

#[test]
fn test_model_iter() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.iters = 1000;
    model.train(&data);

    let a = model.predict(&pred_data);

    assert_eq!(a.data.len(), 3);
}

#[test]
fn test_model_forgy() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::Forgy;
    model.train(&data);

    let a = model.predict(&pred_data);

    assert_eq!(a.data.len(), 3);
}

#[test]
fn test_model_ran_partition() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::RandomPartition;
    model.train(&data);

    let a = model.predict(&pred_data);

    assert_eq!(a.data.len(), 3);
}

#[test]
fn test_model_kplusplus() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::KPlusPlus;
    model.train(&data);

    let a = model.predict(&pred_data);

    assert_eq!(a.data.len(), 3);
}