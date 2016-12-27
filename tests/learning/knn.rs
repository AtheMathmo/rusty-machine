
#[cfg(datasets)]
pub mod tests {

    use rm::linalg::{BaseMatrix, Vector};
    use rm::learning::SupModel;
    use rm::learning::knn::KNNClassifier;
    use rm::datasets::iris;

    #[test]
    fn test_knn() {
        let dataset = iris::load();
        // slice first 2 columns
        let data = dataset.data().select_cols(&[0, 1]);

        let mut knn = KNNClassifier::new(1, 30);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![5.9, 3.6]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(4, 30);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![5.9, 3.6]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(4, 30);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![6.0, 3.5]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(5, 30);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![7.1, 2.8]).unwrap();
        assert_eq!(res, Vector::new(vec![2]));
    }
}
