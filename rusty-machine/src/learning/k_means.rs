use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::UnSupModel;

struct KMeansClassifier {
    iters: usize,
    k: usize,
    centroids: Matrix<f64>,
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier {
    fn predict(&self, data: Matrix<f64>) -> Vector<usize> {
        Vector::new(vec![0])
    }

    fn train(&mut self, data: Matrix<f64>) {}
}

fn init_centroids(kmeans: KMeansClassifier) {}

fn find_closest_centroids(kmeans: KMeansClassifier, data: Matrix<f64>) {}

fn compute_means(kmeans: KMeansClassifier, data: Matrix<f64>, classes: Vector<usize>) {}
