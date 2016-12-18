use rm::linalg::{Matrix, Vector};
use rm::learning::agglomerative::{AgglomerativeClustering, Metrics};

#[test]
fn test_cluster() {
    let data = Matrix::new(7, 5, vec![89., 90., 67. ,46., 50.,
                                      57., 70., 80., 85., 90.,
                                      80., 90., 35., 40., 50.,
                                      40., 60., 50., 45., 55.,
                                      78., 85., 45., 55., 60.,
                                      55., 65., 80., 75., 85.,
                                      90., 85., 88., 92., 95.]);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Single);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Complete);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Average);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Centroid);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Median);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Ward1);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Ward2);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);

    let mut hclust = AgglomerativeClustering::new(3, Metrics::Ward);
    let res = hclust.train(&data);
    let exp = Vector::new(vec![1, 2, 1, 0, 1, 2, 2]);
    assert_eq!(res.unwrap(), exp);
}

