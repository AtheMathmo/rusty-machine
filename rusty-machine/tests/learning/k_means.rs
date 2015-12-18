use rm::linalg::matrix::Matrix;
use rm::linalg::vector::Vector;
use rm::learning::UnSupModel;
use rm::learning::k_means::KMeansClassifier;

#[test]
fn test_model() {
    let mut model = KMeansClassifier::new(3);
    let data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    let pred_data = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    model.train(data);

    let a = model.predict(pred_data);

    assert_eq!(a.data.len(), 3);
}