use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::UnSupModel;

struct KMeansClassifier {
    iters: usize,
    k: usize,
    centroids: Option<Matrix<f64>>,
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier {
    fn predict(&self, data: Matrix<f64>) -> Vector<usize> {
        Vector::new(vec![0])
    }

    fn train(&mut self, data: Matrix<f64>) {
        self.init_centroids(data.cols);

        for i in 0..self.iters {
            let idx = self.find_closest_centroids(&data);
            self.compute_means(&data, idx);
        }
    }
}

impl KMeansClassifier {
    pub fn new(k: usize) -> KMeansClassifier {
        KMeansClassifier {
            iters: 100,
            k: k,
            centroids: None,
        }
    }

    fn init_centroids(&mut self, dim: usize) {
        // These should not all be equal!
        self.centroids = Some(Matrix::zeros(self.k, dim));
    }

    fn find_closest_centroids(&self, data: &Matrix<f64>) -> Vector<usize> {
        let mut idx = Vector::zeros(data.rows);

        match self.centroids {
            Some(ref c) => {
                for i in 0..data.rows {
                    let centroid_dist = c - data.select_rows(&vec![i; c.rows]);
                    let dist = &centroid_dist.elemul(&centroid_dist).sum_cols();
                    // Now take argmin and this is the centroid.

                }
            }
            None => panic!("Centroids not defined."),
        }

        idx
    }

    fn compute_means(&mut self, data: &Matrix<f64>, classes: Vector<usize>) {
        for i in 0..self.k {
            let mut vec_k = Vec::new();

            for j in classes.data.iter() {
                if *j == i {
                    vec_k.push(j);
                }
            }

            // index out these rows from data
            // make a new matrix
            // compute average of it's rows
        }
    }
}
