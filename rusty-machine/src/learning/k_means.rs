//! K-means Classification
//!
//! Provides implementation of K-Means classification.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::learning::k_means::KMeansClassifier;
//! use rusty_machine::learning::UnSupModel;
//!
//! let train_data = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
//! let pred_data = Matrix::new(1,2, vec![0.0, 0.0]);
//! 
//! // Create model with k(=2) classes.
//! let mut model = KMeansClassifier::new(2);
//!
//! // Where train_data is a Matrix with features in columns.
//! model.train(&train_data); 
//!
//! // Where pred_data is a Matrix with features in columns.
//! let a = model.predict(&pred_data);
//! ```
//!
//! Additionally you can control the initialization
//! algorithm and max number of iterations.
//!
//! # Initializations
//!
//! Three initialization algorithms are supported.
//!
//! ## Forgy initialization
//!
//! Choose initial centroids randomly from the data.
//!
//! ## Random Partition initialization
//!
//! Randomly assign each data point to one of k clusters.
//! The initial centroids are the mean of the data in their class.
//!
//! ## K-means++ initialization
//!
//! The [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) scheme.

use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::UnSupModel;
use rand::{Rng, thread_rng};

/// Initialization Algorithm enum.
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
    fn predict(&self, data: &Matrix<f64>) -> Vector<usize> {
        if let Some(ref centroids) = self.centroids {
            return KMeansClassifier::find_closest_centroids(centroids, data).0;
        }
        else {
            panic!("Model has not been trained.");
        }
    }

    /// Train the classifier using input data.
    fn train(&mut self, data: &Matrix<f64>) {
        self.init_centroids(data);
        let mut cost = 0.0;

        for _i in 0..self.iters {
                let (idx, distances) = self.get_closest_centroids(data);
                self.update_centroids(data, idx);

                let cost_i = distances.sum();
                if cost == cost_i {
                    break;
                }

                cost = cost_i;
        }
    }
}

impl KMeansClassifier {
    /// Constructs untrained k-means classifier model.
    ///
    /// Requires number of classes to be specified.
    /// Defaults to 100 iterations and kmeans++ initialization.
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
            init_algorithm: InitAlgorithm::KPlusPlus,
        }
    }

    /// Initialize the centroids.
    ///
    /// Used internally within model.
    fn init_centroids(&mut self, data: &Matrix<f64>) {
        match self.init_algorithm {
            InitAlgorithm::Forgy => {
                self.centroids = Some(KMeansClassifier::forgy_init(self.k, data))
            }
            InitAlgorithm::RandomPartition => {
                self.centroids = Some(KMeansClassifier::ran_partition_init(self.k, data))
            }
            InitAlgorithm::KPlusPlus => {
                self.centroids = Some(KMeansClassifier::plusplus_init(self.k, data))
            }
        }
    }

    /// Updated the centroids by computing means of assigned classes.
    ///
    /// Used internally within model.
    fn update_centroids(&mut self, data: &Matrix<f64>, classes: Vector<usize>) {
        let mut new_centroids = Vec::with_capacity(self.k * data.cols());
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

        self.centroids = Some(Matrix::new(self.k, data.cols(), new_centroids));
    }

    fn get_closest_centroids(&self, data: &Matrix<f64>) -> (Vector<usize>, Vector<f64>) {
        if let Some(ref c) = self.centroids {
            return KMeansClassifier::find_closest_centroids(&c, data);
        }
        else {
            panic!("Centroids not correctly initialized.");
        }
    }

    /// Find the centroid closest to each data point.
    ///
    /// Used internally within model.
    /// Returns the index of the closest centroid and the distance to it.
    fn find_closest_centroids(centroids: &Matrix<f64>, data: &Matrix<f64>) -> (Vector<usize>, Vector<f64>) {
        let mut idx = Vector::zeros(data.rows());
        let mut distances = Vector::zeros(data.rows());

        for i in 0..data.rows() {
            // This works like repmat pulling out row i repeatedly.
            let centroid_diff = centroids - data.select_rows(&vec![i; centroids.rows()]);
            let dist = &centroid_diff.elemul(&centroid_diff).sum_cols();

            // Now take argmin and this is the centroid.
            let (min_idx, min_dist) = dist.argmin();
            idx.data[i]= min_idx;
            distances.data[i] = min_dist;

        }

        (idx, distances)
    }

    /// Compute initial centroids using Forgy scheme.
    ///
    /// Selects k random points in data for centroids.
    fn forgy_init(k: usize, data: &Matrix<f64>) -> Matrix<f64> {
        assert!(k <= data.rows());

        let mut random_choices = Vec::with_capacity(k);
        let mut rng = thread_rng();
        while random_choices.len() < k {
            let r = rng.gen_range(0, data.rows());

            if !random_choices.contains(&r) {
                random_choices.push(r);
            }
        }

        data.select_rows(&random_choices)
    }

    /// Compute initial centroids using random partition.
    ///
    /// Selects centroids by assigning each point randomly to a class
    /// and computing the mean of each class.
    fn ran_partition_init(k: usize, data: &Matrix<f64>) -> Matrix<f64> {
        assert!(k <= data.rows());

        let mut random_assignments = Vec::with_capacity(data.rows());

        // Populate so we have something in each class.
        for i in 0..k {
            random_assignments.push(i);
        }

        let mut rng = thread_rng();
        for _i in k..data.rows() {
            random_assignments.push(rng.gen_range(0, k));
        }

        let mut init_centroids = Vec::with_capacity(k * data.cols());
        for i in 0..k {
            let mut vec_i = Vec::new();

            for j in random_assignments.iter() {
                if *j == i {
                    vec_i.push(*j);
                }
            }

            let mat_i = data.select_rows(&vec_i);
            init_centroids.extend(mat_i.mean(0).data);
        }

        Matrix::new(k, data.cols(), init_centroids)
    }

    /// Compute initial centroids using k-means++.
    ///
    /// Selects centroids using weighted probability from
    /// distances.
    fn plusplus_init(k: usize, data: &Matrix<f64>) -> Matrix<f64> {
        assert!(k <= data.rows());

        let mut rng = thread_rng();

        let mut init_centroids = Vec::with_capacity(k * data.cols());
        let first_cen = rng.gen_range(0usize, data.rows());
        
        init_centroids.append(&mut data.select_rows(&vec![first_cen]).data);

        for i in 1..k {
            let temp_centroids = Matrix::new(i, data.cols(), init_centroids.clone());
            let (_, dist) = KMeansClassifier::find_closest_centroids(&temp_centroids, &data);
            let next_cen = sample_discretely(dist);
            init_centroids.append(&mut data.select_rows(&vec![next_cen]).data)
        }
        
        Matrix::new(k, data.cols(), init_centroids)
    }
}

/// Sample from an unnormalized distribution.
///
/// 
fn sample_discretely(unnorm_dist: Vector<f64>) -> usize {
    assert!(unnorm_dist.size() > 0);

    let sum = unnorm_dist.sum();

    let rand = thread_rng().gen_range(0.0f64, sum);

    let mut tempsum = 0.0;
    for (i,p) in unnorm_dist.data.iter().enumerate() {
        tempsum += *p;

        if rand < tempsum {
            return i;
        }
    }

    panic!("No random value was sampled! There may be more clusters than unique data points.");
}