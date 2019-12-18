//! - k-Nearest Nerighbors
//!
//! Contains implemention of k-nearest search using
//! kd-tree, ball-tree and brute-force.
//!
//! # Usage
//!
//! ```
//! # #[macro_use] extern crate rulinalg; extern crate rusty_machine; fn main() {
//! use rusty_machine::learning::knn::KNNClassifier;
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::Vector;
//!
//! let data = matrix![1., 1., 1.;
//!                    1., 2., 3.;
//!                    2., 3., 1.;
//!                    2., 2., 0.];
//! let target = Vector::new(vec![0, 0, 1, 1]);
//!
//! // train the model to search 2-nearest
//! let mut knn = KNNClassifier::new(2);
//! knn.train(&data, &target).unwrap();
//!
//! // predict new points
//! let res = knn.predict(&matrix![2., 3., 0.; 1., 1., 2.]).unwrap();
//! assert_eq!(res, Vector::new(vec![1, 0]));
//! # }
//! ```
use std::f64;
use std::collections::BTreeMap;

use linalg::{Matrix, BaseMatrix, Vector};
use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};

mod binary_tree;
mod brute_force;

pub use self::binary_tree::{KDTree, BallTree};
pub use self::brute_force::BruteForce;

/// k-Nearest Neighbor Classifier
#[derive(Debug)]
pub struct KNNClassifier<S: KNearestSearch> {
    k: usize,

    searcher: S,
    target: Option<Vector<usize>>,
}

impl Default for KNNClassifier<KDTree> {
    /// Constructs an untrained KNN Classifier with searching 5 neighbors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::KNNClassifier;
    /// let _ = KNNClassifier::default();
    /// ```
    fn default() -> Self {
        KNNClassifier {
            k: 5,
            searcher: KDTree::default(),
            target: None
        }
    }
}

impl KNNClassifier<KDTree> {
    /// Constructs an untrained KNN Classifier with specified
    /// number of search neighbors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::KNNClassifier;
    /// let _ = KNNClassifier::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        KNNClassifier {
            k: k,
            searcher: KDTree::default(),
            target: None
        }
    }
}

impl<S: KNearestSearch> KNNClassifier<S> {
    /// Constructs an untrained KNN Classifier with specified
    /// k and leafsize for KDTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::{KNNClassifier, BallTree};
    /// let _ = KNNClassifier::new_specified(3, BallTree::new(10));
    /// ```
    pub fn new_specified(k: usize, searcher: S) -> Self {
        KNNClassifier {
            k: k,
            searcher: searcher,
            target: None
        }
    }
}

impl<S: KNearestSearch> SupModel<Matrix<f64>, Vector<usize>> for KNNClassifier<S> {

    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<usize>> {
        match self.target {
            Some(ref target) => {

                let mut results: Vec<usize> = Vec::with_capacity(inputs.rows());
                for row in inputs.row_iter() {
                    let (idx, _) = self.searcher.search(row.raw_slice(), self.k)?;
                    let res = target.select(&idx);
                    let (uniques, counts) = freq(res.data());
                    let (id, _) = counts.argmax();
                    results.push(uniques[id]);
                }
                Ok(Vector::new(results))
            },
            _ => Err(Error::new_untrained())
        }
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<usize>) -> LearningResult<()> {
        if inputs.rows() != targets.size() {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "inputs and targets must be the same length"));
        }
        if inputs.rows() < self.k {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "inputs number of rows must be equal or learger than k"));
        }
        self.searcher.build(inputs.clone());
        self.target = Some(targets.clone());
        Ok(())
    }
}

/// Container for k-Nearest search results
struct KNearest {
    // number to search
    k: usize,
    // tuple of index and its distances, sorted by distances
    pairs: Vec<(usize, f64)>,
}

impl KNearest {

    fn new(k: usize, index: Vec<usize>, distances: Vec<f64>) -> Self {
        debug_assert!(!index.is_empty(), "index can't be empty");
        debug_assert!(index.len() == distances.len(),
                      "index and distance must have the same length");

        let mut pairs: Vec<(usize, f64)> = index.into_iter()
                                                .zip(distances.into_iter())
                                                .collect();
        // sort by distance, take k elements
        pairs.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        pairs.truncate(k);

        KNearest {
            k: k,
            pairs: pairs
        }
    }

    /// Add new index and distances to the container, keeping first k elements which
    /// distances are smaller. Returns the updated farthest distance.
    fn add(&mut self, index: usize, distance: f64) -> f64 {
        // self.pairs can't be empty
        let len = self.pairs.len();
        // index of the last element after the query
        let last_index: usize = if len < self.k {
            len
        } else {
            len - 1
        };

        unsafe {
            if self.pairs.get_unchecked(len - 1).1 < distance {
                if len < self.k {
                    // append to the last
                    self.pairs.push((index, distance));
                }
                self.pairs.get_unchecked(last_index).1
            } else {
                // last element is already compared
                if len >= self.k {
                    self.pairs.pop().unwrap();
                }

                for i in 2..(len + 1) {
                    if self.pairs.get_unchecked(len - i).1 < distance {
                        self.pairs.insert(len - i + 1, (index, distance));
                        return self.pairs.get_unchecked(last_index).1;
                    }
                }
                self.pairs.insert(0, (index, distance));
                self.pairs.get_unchecked(last_index).1
            }
        }
    }

    /// Return the k-th distance with searching point
    fn dist(&self) -> f64 {
        // KNearest should gather k element at least
        let len = self.pairs.len();
        if len < self.k {
            f64::MAX
        } else {
            unsafe {
                // unchecked ver of .last().unwrap(),
                // because self.pairs can't be empty
                self.pairs.get_unchecked(len - 1).1
            }
        }
    }

    /// Extract the search result to k-nearest indices and corresponding distances
    fn get_results(self) -> (Vec<usize>, Vec<f64>) {
        let mut indices: Vec<usize> = Vec::with_capacity(self.k);
        let mut distances: Vec<f64> = Vec::with_capacity(self.k);
        for (i, d) in self.pairs {
            indices.push(i);
            distances.push(d);
        }
        (indices, distances)
    }
}

/// Search K-nearest items
pub trait KNearestSearch: Default{

    /// build data structure for search optimization
    fn build(&mut self, data: Matrix<f64>);

    /// Serch k-nearest items close to the point
    /// Returns a tuple of searched item index and its distances
    fn search(&self, point: &[f64], k: usize) -> Result<(Vec<usize>, Vec<f64>), Error>;
}

/// Count target label frequencies
/// TODO: Used in decisition tree, move impl to somewhere
fn freq(labels: &[usize]) -> (Vector<usize>, Vector<usize>) {
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    for l in labels {
        let e = map.entry(*l).or_insert(0);
        *e += 1;
    }

    let mut uniques: Vec<usize> = Vec::with_capacity(map.len());
    let mut counts: Vec<usize> = Vec::with_capacity(map.len());
    for (&k, &v) in &map {
        uniques.push(k);
        counts.push(v);
    }
    (Vector::new(uniques), Vector::new(counts))
}

/// Return distances between given point and data specified with row ids
fn get_distances(data: &Matrix<f64>, point: &[f64], ids: &[usize]) -> Vec<f64> {
    assert!(!ids.is_empty(), "target ids is empty");

    let mut distances: Vec<f64> = Vec::with_capacity(ids.len());
    for id in ids.iter() {
        // ToDo: use .row(*id)
        let row: Vec<f64> = data.select_rows(&[*id]).into_vec();
        // let row: Vec<f64> = self.data.row(*id).into_vec();
        let d = dist(point, &row);
        distances.push(d);
    }
    distances
}

fn dist(v1: &[f64], v2: &[f64]) -> f64 {
    // ToDo: use metrics
    let d: f64 = v1.iter()
                   .zip(v2.iter())
                   .map(|(&x, &y)| (x - y) * (x - y))
                   .fold(0., |s, v| s + v);
    d.sqrt()
}

#[cfg(test)]
mod tests {

    use std::f64;
    use super::KNearest;

    #[test]
    fn test_knearest() {
        let mut kn = KNearest::new(2, vec![1, 2, 3], vec![3., 2., 1.]);
        assert_eq!(kn.k, 2);
        assert_eq!(kn.pairs, vec![(3, 1.), (2, 2.)]);
        assert_eq!(kn.dist(), 2.);

        // update KNearest
        let res = kn.add(10, 3.);
        assert_eq!(res, 2.);
        assert_eq!(kn.k, 2);
        assert_eq!(kn.pairs, vec![(3, 1.), (2, 2.)]);
        assert_eq!(kn.dist(), 2.);

        let res = kn.add(11, 0.);
        assert_eq!(res, 1.);
        assert_eq!(kn.k, 2);
        assert_eq!(kn.pairs, vec![(11, 0.), (3, 1.)]);
        assert_eq!(kn.dist(), 1.);
    }

    #[test]
    fn test_knearest2() {
        let mut kn = KNearest::new(4, vec![1, 2, 3], vec![3., 2., 1.]);
        assert_eq!(kn.k, 4);
        assert_eq!(kn.pairs, vec![(3, 1.), (2, 2.), (1, 3.)]);
        assert_eq!(kn.dist(), f64::MAX);

        let res = kn.add(5, 1.5);
        assert_eq!(res, 3.);
        assert_eq!(kn.k, 4);
        assert_eq!(kn.pairs, vec![(3, 1.), (5, 1.5), (2, 2.), (1, 3.)]);
        assert_eq!(kn.dist(), 3.);

        let res = kn.add(6, 6.);
        assert_eq!(res, 3.);
        assert_eq!(kn.k, 4);
        assert_eq!(kn.pairs, vec![(3, 1.), (5, 1.5), (2, 2.), (1, 3.)]);
        assert_eq!(kn.dist(), 3.);

        let res = kn.add(7, 0.5);
        assert_eq!(res, 2.);
        assert_eq!(kn.k, 4);
        assert_eq!(kn.pairs, vec![(7, 0.5), (3, 1.), (5, 1.5), (2, 2.)]);
        assert_eq!(kn.dist(), 2.);
    }
}
