use rm::linalg::Matrix;

use rm::learning::dbscan::DBSCAN;
use rm::learning::UnSupModel;

#[test]
fn test_basic_clusters() {
    let inputs = Matrix::new(6, 2, vec![1.0, 2.0,
                                        1.1, 2.2,
                                        0.9, 1.9,
                                        1.0, 2.1,
                                        -2.0, 3.0,
                                        -2.2, 3.1]);

    let mut model = DBSCAN::new(0.5, 2);
    model.train(&inputs);

    let clustering = model.clusters().unwrap();

    assert!(clustering.data().iter().take(4).all(|x| *x == Some(0)));
    assert!(clustering.data().iter().skip(4).all(|x| *x == Some(1)));
}


#[test]
fn test_basic_prediction() {
    let inputs = Matrix::new(6, 2, vec![1.0, 2.0,
                                        1.1, 2.2,
                                        0.9, 1.9,
                                        1.0, 2.1,
                                        -2.0, 3.0,
                                        -2.2, 3.1]);

    let mut model = DBSCAN::new(0.5, 2);
    model.set_predictive(true);
    model.train(&inputs);

    let new_points = Matrix::new(2,2, vec![1.0, 2.0, 4.0, 4.0]);

    let classes = model.predict(&new_points);
    assert!(classes[0] == Some(0));
    assert!(classes[1] == None);
}
