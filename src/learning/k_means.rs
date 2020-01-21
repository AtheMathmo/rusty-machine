//! K-means Classification
//!
//! Provides implementation of K-Means classification.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::linalg::Matrix;
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
//! model.train(&inputs).unwrap();
//!
//! // Where test_inputs is a Matrix with features in columns.
//! let a = model.predict(&test_inputs).unwrap();
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

use linalg::{Matrix, MatrixSlice, Axes, Vector, BaseMatrix};
use learning::{LearningResult, UnSupModel};
use learning::error::{Error, ErrorKind};

use rand::{Rng, thread_rng};
use libnum::abs;

use std::fmt::Debug;

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
pub struct KMeansClassifier<InitAlg: Initializer> {
    /// Max iterations of algorithm to run.
    iters: usize,
    /// The number of classes.
    k: usize,
    /// The fitted centroids .
    centroids: Option<Matrix<f64>>,
    /// The initial algorithm to use.
    init_algorithm: InitAlg,
}

impl<InitAlg: Initializer> UnSupModel<Matrix<f64>, Vector<usize>> for KMeansClassifier<InitAlg> {
    /// Predict classes from data.
    ///
    /// Model must be trained.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<usize>> {
        if let Some(ref centroids) = self.centroids {
            Ok(KMeansClassifier::<InitAlg>::find_closest_centroids(centroids.as_slice(), inputs).0)
        } else {
            Err(Error::new_untrained())
        }
    }

    /// Train the classifier using input data.
    fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        self.init_centroids(inputs)?;
        let mut cost = 0.0;
        let eps = 1e-14;

        for _i in 0..self.iters {
            let (idx, distances) = self.get_closest_centroids(inputs)?;
            self.update_centroids(inputs, idx);

            let cost_i = distances.sum();
            if abs(cost - cost_i) < eps {
                break;
            }

            cost = cost_i;
        }

        Ok(())
    }
}

impl KMeansClassifier<KPlusPlus> {
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
    pub fn new(k: usize) -> KMeansClassifier<KPlusPlus> {
        KMeansClassifier {
            iters: 100,
            k: k,
            centroids: None,
            init_algorithm: KPlusPlus,
        }
    }
}

impl<InitAlg: Initializer> KMeansClassifier<InitAlg> {
    /// Constructs untrained k-means classifier model.
    ///
    /// Requires number of classes, number of iterations, and
    /// the initialization algorithm to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::k_means::{KMeansClassifier, Forgy};
    ///
    /// let model = KMeansClassifier::new_specified(5, 42, Forgy);
    /// ```
    pub fn new_specified(k: usize, iters: usize, algo: InitAlg) -> KMeansClassifier<InitAlg> {
        KMeansClassifier {
            iters: iters,
            k: k,
            centroids: None,
            init_algorithm: algo,
        }
    }

    /// Get the number of classes.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the number of iterations.
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// Get the initialization algorithm.
    pub fn init_algorithm(&self) -> &InitAlg {
        &self.init_algorithm
    }

    /// Get the centroids `Option<Matrix<f64>>`.
    pub fn centroids(&self) -> &Option<Matrix<f64>> {
        &self.centroids
    }

    /// Set the number of iterations.
    pub fn set_iters(&mut self, iters: usize) {
        self.iters = iters;
    }

    /// Initialize the centroids.
    ///
    /// Used internally within model.
    fn init_centroids(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        if self.k > inputs.rows() {
            Err(Error::new(ErrorKind::InvalidData,
                           format!("Number of clusters ({0}) exceeds number of data points \
                                    ({1}).",
                                   self.k,
                                   inputs.rows())))
        } else {
            let centroids = self.init_algorithm.init_centroids(self.k, inputs)?;

            if centroids.rows() != self.k {
                Err(Error::new(ErrorKind::InvalidState,
                                    "Initial centroids must have exactly k rows."))
            } else if centroids.cols() != inputs.cols() {
                Err(Error::new(ErrorKind::InvalidState,
                                    "Initial centroids must have the same column count as inputs."))
            } else {
                self.centroids = Some(centroids);
                Ok(())
            }
        }

    }

    /// Updated the centroids by computing means of assigned classes.
    ///
    /// Used internally within model.
    fn update_centroids(&mut self, inputs: &Matrix<f64>, classes: Vector<usize>) {
        let mut new_centroids = Vec::with_capacity(self.k * inputs.cols());

        let mut row_indexes = vec![Vec::new(); self.k];
        for (i, c) in classes.into_vec().into_iter().enumerate() {
            row_indexes.get_mut(c as usize).map(|v| v.push(i));
        }

        for vec_i in row_indexes {
            let mat_i = inputs.select_rows(&vec_i);
            new_centroids.extend(mat_i.mean(Axes::Row).into_vec());
        }

        self.centroids = Some(Matrix::new(self.k, inputs.cols(), new_centroids));
    }

    fn get_closest_centroids(&self,
                             inputs: &Matrix<f64>)
                             -> LearningResult<(Vector<usize>, Vector<f64>)> {
        if let Some(ref c) = self.centroids {
            Ok(KMeansClassifier::<InitAlg>::find_closest_centroids(c.as_slice(), inputs))
        } else {
            Err(Error::new(ErrorKind::InvalidState,
                           "Centroids not correctly initialized."))
        }
    }

    /// Find the centroid closest to each data point.
    ///
    /// Used internally within model.
    /// Returns the index of the closest centroid and the distance to it.
    fn find_closest_centroids(centroids: MatrixSlice<f64>,
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
}

/// Trait for algorithms initializing the K-means centroids.
pub trait Initializer: Debug {
    /// Initialize the centroids for the initial state of the K-Means model.
    ///
    /// The `Matrix` returned must have `k` rows and the same column count as `inputs`.
    fn init_centroids(&self, k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>>;
}

/// The Forgy initialization scheme.
#[derive(Debug)]
pub struct Forgy;

impl Initializer for Forgy {
    fn init_centroids(&self, k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
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

/// The Random Partition initialization scheme.
#[derive(Debug)]
pub struct RandomPartition;

impl Initializer for RandomPartition {
    fn init_centroids(&self, k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {

        // Populate so we have something in each class.
        let mut random_assignments = (0..k).map(|i| vec![i]).collect::<Vec<Vec<usize>>>();
        let mut rng = thread_rng();
        for i in k..inputs.rows() {
            let idx = rng.gen_range(0, k);
            unsafe {
                random_assignments.get_unchecked_mut(idx).push(i);
            }
        }

        let mut init_centroids = Vec::with_capacity(k * inputs.cols());

        for vec_i in random_assignments {
            let mat_i = inputs.select_rows(&vec_i);
            init_centroids.extend_from_slice(&*mat_i.mean(Axes::Row).into_vec());
        }

        Ok(Matrix::new(k, inputs.cols(), init_centroids))
    }
}

/// The K-means ++ initialization scheme.
#[derive(Debug)]
pub struct KPlusPlus;

impl Initializer for KPlusPlus {
    fn init_centroids(&self, k: usize, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        let mut rng = thread_rng();

        let mut init_centroids = Vec::with_capacity(k * inputs.cols());
        let first_cen = rng.gen_range(0usize, inputs.rows());

        unsafe {
            init_centroids.extend_from_slice(inputs.row_unchecked(first_cen).raw_slice());
        }

        for i in 1..k {
            unsafe {
                let temp_centroids = MatrixSlice::from_raw_parts(init_centroids.as_ptr(),
                                                                 i,
                                                                 inputs.cols(),
                                                                 inputs.cols());
                let (_, dist) =
                    KMeansClassifier::<KPlusPlus>::find_closest_centroids(temp_centroids, inputs);

                // A relatively cheap way to validate our input data
                if !dist.data().iter().all(|x| x.is_finite()) {
                    return Err(Error::new(ErrorKind::InvalidData,
                                          "Input data led to invalid centroid distances during \
                                           initialization."));
                }

                let next_cen = sample_discretely(&dist);
                init_centroids.extend_from_slice(inputs.row_unchecked(next_cen).raw_slice());
            }
        }

        Ok(Matrix::new(k, inputs.cols(), init_centroids))
    }
}

/// Sample from an unnormalized distribution.
///
/// The input to this function is assumed to have all positive entries.
fn sample_discretely(unnorm_dist: &Vector<f64>) -> usize {
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
