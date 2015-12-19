//! K Means Classification
//!
//! 
//!
//!


use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::UnSupModel;
use rand;
use rand::Rng;

pub enum InitAlgorithm {
    Forgy,
    RandomPartition,
    KPlusPlus,
}

/// K-Means Classification model.
///
/// Contains option for centroids.
/// Specifies iterations and number of classes.
pub struct KMeansClassifier {
    pub iters: usize,
    pub k: usize,
    pub centroids: Option<Matrix<f64>>,
    pub init_algorithm: InitAlgorithm,
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier {

    /// Predict classes from data.
    ///
    /// Model must be trained.
    fn predict(&self, data: Matrix<f64>) -> Vector<usize> {
        match self.centroids {
            Some(ref _c) => return self.find_closest_centroids(&data),
            None => panic!("Model has not been trained."),
        }
        
    }

    /// Train the classifier using input data.
    fn train(&mut self, data: Matrix<f64>) {
        self.init_centroids(&data);

        for _i in 0..self.iters {
            let idx = self.find_closest_centroids(&data);
            self.update_centroids(&data, idx);
        }
    }
}

impl KMeansClassifier {

    /// Constructs untrained k-means classifier model.
    ///
    /// Requires number of classes to be specified.
    /// Defaults to 100 iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifier;
    ///
    /// let model = KMeansClassifier::new(5);
    /// ```
    pub fn new(k: usize) -> KMeansClassifier {
        KMeansClassifier {
            iters: 100,
            k: k,
            centroids: None,
            init_algorithm: InitAlgorithm::Forgy,
        }
    }

    /// Initialize the centroids.
    ///
    /// Used internally within model.
    /// Currently only supports Forgy initialization.
    fn init_centroids(&mut self, data: &Matrix<f64>) {
        match self.init_algorithm {
            InitAlgorithm::Forgy => self.centroids = Some(forgy_init(self.k, data)),
            _ => self.centroids = Some(forgy_init(self.k, data)),
        }
    }

    /// Find the centroid closest to each data point.
    ///
    /// Used internally within model.
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

    /// Updated the centroids by computing means of assigned classes.
    ///
    /// Used internally within model.
    fn update_centroids(&mut self, data: &Matrix<f64>, classes: Vector<usize>) {
        let mut new_centroids = Vec::with_capacity(self.k * data.cols);
        for i in 0..self.k {
            let mut vec_i = Vec::new();

            for j in classes.data.iter() {
                if *j == i {
                    vec_i.push(*j);
                }
            }

            let mat_i = data.select_rows(&vec_i);
            new_centroids.extend(mat_i.mean(0).data);
        }
        
        self.centroids = Some(Matrix::new(self.k, data.cols, new_centroids));
    }
}

/// Compute initial centroids using Forgy scheme.
///
/// Selects k random points in data for centroids.
fn forgy_init(k: usize, data: &Matrix<f64>) -> Matrix<f64> {
	let mut random_choices = Vec::with_capacity(k);
	while random_choices.len() < k {
		let r = rand::thread_rng().gen_range(0, data.rows);

		if !random_choices.contains(&r) {
			random_choices.push(r);
		}
	}

	data.select_rows(&random_choices)
}
