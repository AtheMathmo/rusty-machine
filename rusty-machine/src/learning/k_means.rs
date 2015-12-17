use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::UnSupModel;

pub struct KMeansClassifier {
    iters: usize,
    k: usize,
    centroids: Option<Matrix<f64>>,
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier {
    fn predict(&self, data: Matrix<f64>) -> Vector<usize> {
        match self.centroids {
            Some(ref _c) => return self.find_closest_centroids(&data),
            None => panic!("Model has not been trained."),
        }
        
    }

    fn train(&mut self, data: Matrix<f64>) {
        self.init_centroids(data.cols);

        for _i in 0..self.iters {
            let idx = self.find_closest_centroids(&data);
            self.update_centroids(&data, idx);
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
                    // This works like repmat pulling out row i repeatedly.
                    let centroid_diff = c - data.select_rows(&vec![i; c.rows]);
                    let dist = &centroid_diff.elemul(&centroid_diff).sum_cols();

                    // Now take argmin and this is the centroid.
                    idx.data[i] = dist.argmin();

                }
            }
            None => panic!("Centroids not defined."),
        }

        idx
    }

    fn update_centroids(&mut self, data: &Matrix<f64>, classes: Vector<usize>) {
        let mut new_centroids = Vec::with_capacity(self.k * classes.size);
        for i in 0..self.k {
            let mut vec_i = Vec::with_capacity(classes.size);

            for j in classes.data.iter() {
                if *j == i {
                    vec_i.push(*j);
                }
            }

            let mat_i = data.select_rows(&vec_i);

            new_centroids.extend(mat_i.mean(0).data);
        }

        self.centroids = Some(Matrix::new(self.k, classes.size, new_centroids));
    }
}
