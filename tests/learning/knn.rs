use rm::linalg::{Matrix, Vector};
use rm::learning::SupModel;
use rm::learning::knn::KNNClassifier;

#[test]
fn test_knn() {
    let data = matrix![1., 1., 1.;
                       1., 2., 3.;
                       2., 3., 1.;
                       2., 2., 0.];
    let target = Vector::new(vec![0, 0, 1, 1]);

    let mut knn = KNNClassifier::new(2);
    let _ = knn.train(&data, &target).unwrap();

    let res = knn.predict(&matrix![2., 3., 0.; 1., 1., 2.]).unwrap();
    let exp = Vector::new(vec![1, 0]);
    assert_eq!(res, exp);
}

#[test]
fn test_knn_long() {
    let vals = (0..200000).map(|x: usize| x as f64).collect::<Vec<f64>>();
    let data = Matrix::new(100000, 2, vals);

    let mut tvals = vec![0; 50000];
    tvals.extend(vec![1; 50000]);
    let target = Vector::new(tvals);

    // check stack doesn't overflow
    let mut knn = KNNClassifier::new(10);
    let _ = knn.train(&data, &target).unwrap();

    let res = knn.predict(&matrix![5., 10.; 60000., 550000.]).unwrap();
    let exp = Vector::new(vec![0, 1]);
    assert_eq!(res, exp);

    // check stack doesn't overflow
    let mut knn = KNNClassifier::new(1000);
    let _ = knn.train(&data, &target).unwrap();
    assert_eq!(res, exp);
}

#[cfg(feature = "datasets")]
pub mod tests_datasets {

    use rm::linalg::{BaseMatrix, Vector};
    use rm::learning::SupModel;
    use rm::learning::knn::{KNNClassifier, KDTree, BallTree, BruteForce};
    use rm::datasets::iris;

    #[test]
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

    #[test]
    fn test_knn_iris_default() {
        let dataset = iris::load();

        let mut knn = KNNClassifier::default();
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();

        let exp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        assert_eq!(res, Vector::new(exp));
    }

    #[test]
    fn test_knn_iris_different_neighbors() {
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

    #[test]
    fn test_knn_iris_new_specified() {
        let dataset = iris::load();

        let mut knn = KNNClassifier::new_specified(5, KDTree::default());
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();

        let exp = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let expv = Vector::new(exp);
        assert_eq!(res, expv);

        let mut knn = KNNClassifier::new_specified(5, BallTree::default());
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();
        assert_eq!(res, expv);

        let mut knn = KNNClassifier::new_specified(5, BruteForce::default());
        let _ = knn.train(&dataset.data(), &dataset.target()).unwrap();
        let res = knn.predict(&dataset.data()).unwrap();
        assert_eq!(res, expv);
    }
}