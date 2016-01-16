use rm::linalg::matrix::Matrix;
use rm::learning::UnSupModel;
use rm::learning::k_means::KMeansClassifier;
use rm::learning::k_means::InitAlgorithm;

#[test]
fn test_model_default() {
    let mut model = KMeansClassifier::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_iter() {
    let mut model = KMeansClassifier::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.iters = 1000;
    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_forgy() {
    let mut model = KMeansClassifier::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::Forgy;
    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_ran_partition() {
    let mut model = KMeansClassifier::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::RandomPartition;
    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_kplusplus() {
    let mut model = KMeansClassifier::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.init_algorithm = InitAlgorithm::KPlusPlus;
    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}
