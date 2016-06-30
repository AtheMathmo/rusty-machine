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
//! let inputs = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
//! let test_inputs = Matrix::new(1, 2, vec![1.0, 3.5]);
//!
//! // Create model with k(=2) classes.
//! let mut model = KMeansClassifier::new(2);
//!
//! // Where inputs is a Matrix with features in columns.
//! model.train(&inputs);
//!
//! // Where test_inputs is a Matrix with features in columns.
//! let a = model.predict(&test_inputs);
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
use learning::error::{Error, ErrorKind};

use rand::{Rng, thread_rng};
use libnum::abs;

/// Initialization Algorithm enum.
#[derive(Clone, Copy, Debug)]
pub enum InitAlgorithm {
    /// The Forgy initialization scheme.
    Forgy,
    /// The Random Partition initialization scheme.
    RandomPartition,
    /// The K-means ++ initialization scheme.
    KPlusPlus,
}

/// K-Means Classification Builder.
#[derive(Debug)]
pub struct KMeansClassifierBuilder {
    k: usize,
    iters: usize,
    init_algorithm: InitAlgorithm
}

impl KMeansClassifierBuilder {
    /// Create a K-Means Classification Builder.
    ///
    /// Requires number of classes to be specified.
    /// Defaults to 100 iterations and kmeans++ initialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifierBuilder;
    ///
    /// // Creates a new KMeansClassifier with 5 clusters
    /// let model = KMeansClassifierBuilder::new(5).finalize();
    /// ```
    pub fn new(k: usize) -> Self {
        KMeansClassifierBuilder {
            k: k,
            iters: 100,
            init_algorithm: InitAlgorithm::KPlusPlus
        }
    }

    /// Changes the number of iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifierBuilder;
    ///
    /// // Creates a new KMeansClassifier with 5 clusters and 42 iterations
    /// let model = KMeansClassifierBuilder::new(5).iters(42).finalize();
    /// ```
    pub fn iters(&mut self, iters: usize) -> &mut KMeansClassifierBuilder {
        self.iters = iters;
        self
    }

    /// Changes the initialization algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifierBuilder;
    /// use rusty_machine::learning::k_means::InitAlgorithm;
    ///
    /// // Creates a new KMeansClassifier with 5 clusters and uses
    /// // the Forgy initialization algorithm
    /// let model = KMeansClassifierBuilder::new(5).init_algorithm(InitAlgorithm::Forgy)
    ///                                            .finalize();
    /// ```
    pub fn init_algorithm(&mut self, init_algorithm: InitAlgorithm) -> &mut KMeansClassifierBuilder {
        self.init_algorithm = init_algorithm;
        self
    }

    /// Returns the KMeansClassifier with the options previously specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifierBuilder;
    /// use rusty_machine::learning::k_means::InitAlgorithm;
    ///
    /// // Creates a new KMeansClassifier with 5 clusters, 42 iterations
    /// // using the Forgy initialization algorithm
    /// let model = KMeansClassifierBuilder::new(5).iters(42)
    ///                                            .init_algorithm(InitAlgorithm::Forgy)
    ///                                            .finalize();
    /// ```
    pub fn finalize(&self) -> KMeansClassifier {
        KMeansClassifier {
            k: self.k,
            iters: self.iters,
            centroids: None,
            init_algorithm: self.init_algorithm
        }
    }
}

/// K-Means Classification model.
///
/// Contains option for centroids.
/// Specifies iterations and number of classes.
///
/// # Usage
///
/// This model is used through the `UnSupModel` trait. The model is
/// trained via the `train` function with a matrix containing rows of
/// feature vectors.
///
/// The model will not check to ensure the data coming in is all valid.
/// This responsibility lies with the user (for now).
#[derive(Debug)]
pub struct KMeansClassifier {
    /// Max iterations of algorithm to run.
    pub iters: usize,
    /// The number of classes.
    pub k: usize,
    /// The fitted centroids .
    pub centroids: Option<Matrix<f64>>,
    /// The initial algorithm to use.
    pub init_algorithm: InitAlgorithm,
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier {
    /// Predict classes from data.
    ///
    /// Model must be trained.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<usize> {
        if let Some(ref centroids) = self.centroids {
            return KMeansClassifier::find_closest_centroids(centroids, inputs).0;
        } else {
            panic!("Model has not been trained.");
        }
    }

    /// Train the classifier using input data.
    fn train(&mut self, inputs: &Matrix<f64>) {
        self.init_centroids(inputs);
        let mut cost = 0.0;
        let eps = 1e-14;

        for _i in 0..self.iters {
            let (idx, distances) = self.get_closest_centroids(inputs);
            self.update_centroids(inputs, idx);

                let cost_i = distances.sum();
                if abs(cost - cost_i) < eps {
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

    /// Constructs untrained k-means classifier model.
    ///
    /// Requires number of classes, number of iterations, and
    /// the initialization algorithm to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::KMeansClassifier;
    /// use rusty_machine::learning::k_means::InitAlgorithm;
    ///
    /// let model = KMeansClassifier::new_specified(5, 42, InitAlgorithm::Forgy);
    /// ```
    pub fn new_specified(k: usize, iters: usize,
                         algo: InitAlgorithm) -> KMeansClassifier {
        KMeansClassifier {
            iters: iters,
            k: k,
            centroids: None,
            init_algorithm: algo,
        }
    }

    /// Get the number of classes
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the number of iterations
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// Get the initialization algorithm
    pub fn init_algorithm(&self) -> InitAlgorithm {
        self.init_algorithm
    }

    /// Initialize the centroids.
    ///
    /// Used internally within model.
    fn init_centroids(&mut self, inputs: &Matrix<f64>) {
        match self.init_algorithm {
            InitAlgorithm::Forgy => {
                self.centroids = Some(KMeansClassifier::forgy_init(self.k, inputs).unwrap())
            }
            InitAlgorithm::RandomPartition => {
                self.centroids = Some(KMeansClassifier::ran_partition_init(self.k, inputs).unwrap())
            }
            InitAlgorithm::KPlusPlus => {
                self.centroids = Some(KMeansClassifier::plusplus_init(self.k, inputs).unwrap())
            }
        }
    }

    /// Updated the centroids by computing means of assigned classes.
    ///
    /// Used internally within model.
    fn update_centroids(&mut self, inputs: &Matrix<f64>, classes: Vector<usize>) {
        let mut new_centroids = Vec::with_capacity(self.k * inputs.cols());
        for i in 0..self.k {
            let mut vec_i = Vec::new();

            for j in classes.data().iter() {
                if *j == i {
                    vec_i.push(*j);
                }
            }

            let mat_i = inputs.select_rows(&vec_i);
            new_centroids.extend(mat_i.mean(0).data());
        }

        self.centroids = Some(Matrix::new(self.k, inputs.cols(), new_centroids));
    }

    fn get_closest_centroids(&self, inputs: &Matrix<f64>) -> (Vector<usize>, Vector<f64>) {
        if let Some(ref c) = self.centroids {
            return KMeansClassifier::find_closest_centroids(&c, inputs);
        } else {
            panic!("Centroids not correctly initialized.");
        }
    }

    /// Find the centroid closest to each data point.
    ///
    /// Used internally within model.
    /// Returns the index of the closest centroid and the distance to it.
    fn find_closest_centroids(centroids: &Matrix<f64>,
                              inputs: &Matrix<f64>)
                              -> (Vector<usize>, Vector<f64>) {
        let mut idx = Vec::with_capacity(inputs.rows());
        let mut distances = Vec::with_capacity(inputs.rows());

        for i in 0..inputs.rows() {
            // This works like repmat pulling out row i repeatedly.
            let centroid_diff = centroids - inputs.select_rows(&vec![i; centroids.rows()]);
            let dist = &centroid_diff.elemul(&centroid_diff).sum_cols();

            // Now take argmin and this is the centroid.
            let (min_idx, min_dist) = dist.argmin();
            idx.push(min_idx);
            distances.push(min_dist);

        }

        (Vector::new(idx), Vector::new(distances))
    }

    /// Compute initial centroids using Forgy scheme.
    ///
    /// Selects k random points in data for centroids.
    fn forgy_init(k: usize, inputs: &Matrix<f64>) -> Result<Matrix<f64>, Error> {
        if k <= inputs.rows() {
            Err(Error::new(ErrorKind::InvalidData, format!("Number of clusters ({0}) exceeds number of data points ({1}).", k, inputs.rows())))
        } else {
            let mut random_choices = Vec::with_capacity(k);
            let mut rng = thread_rng();
            while random_choices.len() < k {
                let r = rng.gen_range(0, inputs.rows());

                if !random_choices.contains(&r) {
                    random_choices.push(r);
                }
            }

            Ok(inputs.select_rows(&random_choices))
        }  
    }

    /// Compute initial centroids using random partition.
    ///
    /// Selects centroids by assigning each point randomly to a class
    /// and computing the mean of each class.
    fn ran_partition_init(k: usize, inputs: &Matrix<f64>) -> Result<Matrix<f64>, Error> {
        if k <= inputs.rows() {
            Err(Error::new(ErrorKind::InvalidData, format!("Number of clusters ({0}) exceeds number of data points ({1}).", k, inputs.rows())))
        } else {
            let mut random_assignments = Vec::with_capacity(inputs.rows());

            // Populate so we have something in each class.
            for i in 0..k {
                random_assignments.push(i);
            }

            let mut rng = thread_rng();
            for _ in k..inputs.rows() {
                random_assignments.push(rng.gen_range(0, k));
            }

            let mut init_centroids = Vec::with_capacity(k * inputs.cols());
            for i in 0..k {
                let mut vec_i = Vec::new();

                for j in &random_assignments {
                    if *j == i {
                        vec_i.push(*j);
                    }
                }

                let mat_i = inputs.select_rows(&vec_i);
                init_centroids.extend(mat_i.mean(0).into_vec());
            }

            Ok(Matrix::new(k, inputs.cols(), init_centroids))
        }  
    }

    /// Compute initial centroids using k-means++.
    ///
    /// Selects centroids using weighted probability from
    /// distances.
    fn plusplus_init(k: usize, inputs: &Matrix<f64>) -> Result<Matrix<f64>, Error> {
        if k <= inputs.rows() {
            Err(Error::new(ErrorKind::InvalidData, format!("Number of clusters ({0}) exceeds number of data points ({1}).", k, inputs.rows())))
        } else {
            let mut rng = thread_rng();

            let mut init_centroids = Vec::with_capacity(k * inputs.cols());
            let first_cen = rng.gen_range(0usize, inputs.rows());

            init_centroids.append(&mut inputs.select_rows(&[first_cen]).into_vec());

            for i in 1..k {
                let temp_centroids = Matrix::new(i, inputs.cols(), init_centroids.clone());
                let (_, dist) = KMeansClassifier::find_closest_centroids(&temp_centroids, &inputs);
                let next_cen = sample_discretely(dist);
                init_centroids.append(&mut inputs.select_rows(&[next_cen]).into_vec())
            }

            Ok(Matrix::new(k, inputs.cols(), init_centroids))
        }

        
    }
}

/// Sample from an unnormalized distribution.
///
/// The input to this function is assumed to have all positive entries.
fn sample_discretely(unnorm_dist: Vector<f64>) -> usize {
    assert!(unnorm_dist.size() > 0, "No entries in distribution vector.");

    let sum = unnorm_dist.sum();

    let rand = thread_rng().gen_range(0.0f64, sum);

    let mut tempsum = 0.0;
    for (i, p) in unnorm_dist.data().iter().enumerate() {
        tempsum += *p;

        if rand < tempsum {
            return i;
        }
    }

    panic!("No random value was sampled! There may be more clusters than unique data points.");
}
