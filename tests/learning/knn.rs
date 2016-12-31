
#[cfg(feature = "datasets")]
pub mod tests_datasets {

    use rm::linalg::{BaseMatrix, Vector};
    use rm::learning::SupModel;
    use rm::learning::knn::KNNClassifier;
    use rm::datasets::iris;

    fn test_knn_iris_2cols() {
        let dataset = iris::load();
        // slice first 2 columns
        let data = dataset.data().select_cols(&[0, 1]);

        let mut knn = KNNClassifier::new(1);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![5.9, 3.6]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(4);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![5.9, 3.6]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(4);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![6.0, 3.5]).unwrap();
        assert_eq!(res, Vector::new(vec![1]));

        let mut knn = KNNClassifier::new(5);
        let _ = knn.train(&data, &dataset.target()).unwrap();
        let res = knn.predict(&matrix![7.1, 2.8]).unwrap();
        assert_eq!(res, Vector::new(vec![2]));
    }

    fn test_knn_iris() {
        let dataset = iris::load();

        let mut knn = KNNClassifier::new(3);
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();

        let exp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        assert_eq!(res, Vector::new(exp));

        let mut knn = KNNClassifier::new(10);
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();

        let exp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        assert_eq!(res, Vector::new(exp));
    }
}
