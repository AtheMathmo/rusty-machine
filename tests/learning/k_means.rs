use rm::linalg::Matrix;
use rm::learning::UnSupModel;
use rm::learning::k_means::KMeansClassifier;
use rm::learning::k_means::{Forgy, RandomPartition, KPlusPlus};

#[test]
fn test_model_default() {
    let mut model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs).unwrap();

    let outputs = model.predict(&targets).unwrap();

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_iter() {
    let mut model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.set_iters(1000);
    model.train(&inputs).unwrap();

    let outputs = model.predict(&targets).unwrap();

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_forgy() {
    let mut model = KMeansClassifier::new_specified(3, 100, Forgy);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs).unwrap();

    let outputs = model.predict(&targets).unwrap();

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_ran_partition() {
    let mut model = KMeansClassifier::new_specified(3, 100, RandomPartition);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs).unwrap();

    let outputs = model.predict(&targets).unwrap();

    assert_eq!(outputs.size(), 3);
}

#[test]
fn test_model_kplusplus() {
    let mut model = KMeansClassifier::new_specified(3, 100, KPlusPlus);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let targets = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(&inputs).unwrap();

    let outputs = model.predict(&targets).unwrap();

    assert_eq!(outputs.size(), 3);
}

#[test]
#[should_panic]
fn test_no_train_predict() {
    let model = KMeansClassifier::<KPlusPlus>::new(3);
    let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.predict(&inputs).unwrap();

}

#[test]
fn test_two_centroids() {
    let mut model = KMeansClassifier::new(2);
    let inputs = Matrix::new(6, 2, vec![59.59375, 270.6875,
                                        51.59375, 307.6875,
                                        86.59375, 286.6875,
                                        319.59375, 145.6875,
                                        314.59375, 174.6875,
                                        350.59375, 161.6875]);

    model.train(&inputs).unwrap();

    let classes = model.predict(&inputs).unwrap();
    let class_a = classes[0];

    let class_b = if class_a == 0 { 1 } else { 0 };

    assert!(classes.data().iter().take(3).all(|x| *x == class_a));
    assert!(classes.data().iter().skip(3).all(|x| *x == class_b));
}
