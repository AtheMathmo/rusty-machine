use rm::linalg::Matrix;
use rm::learning::UnSupModel;
use rm::learning::k_means::KMeansClassifier;
use rm::learning::k_means::{Forgy, RandomPartition, KPlusPlus};

#[test]
fn test_model_default() {
    let mut model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_iter() {
    let mut model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.set_iters(1000);
    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_forgy() {
    let mut model = KMeansClassifier::new_specified(3, 100, Forgy);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_ran_partition() {
    let mut model = KMeansClassifier::new_specified(3, 100, RandomPartition);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_kplusplus() {
    let mut model = KMeansClassifier::new_specified(3, 100, KPlusPlus);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs);

    let outputs = model.predict(&targets);

    assert_eq!(outputs.size(), 3);
}

#[test]
#[should_panic]
fn test_no_train_predict() {
    let model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.predict(&inputs);

}