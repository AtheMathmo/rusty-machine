//! - k-Nearest Nerighbors

use std::f64;
use std::collections::BTreeMap;

use linalg::{Matrix, BaseMatrix, Vector};
use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};

pub mod kdtree;
pub mod bruteforce;

use self::kdtree::KDTree;

/// k-Nearest Neighbor Classifier
#[derive(Debug)]
pub struct KNNClassifier {
    k: usize,
    leafsize: usize,

    // set after train
    tree: Option<KDTree>,
    target: Option<Vector<usize>>,
}

impl KNNClassifier {

    /// Constructs an untrained KNN Classifier with specified
    /// k and leafsize for KDTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::KNNClassifier;
    /// let _ = KNNClassifier::new(3, 30);
    /// ```
    pub fn new(k: usize, leafsize: usize) -> Self {
        KNNClassifier {
            k: k,
            leafsize: leafsize,

            tree: None,
            target: None
        }
    }
}

impl<'a> SupModel<Matrix<f64>, Vector<usize>> for KNNClassifier {

    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<usize>> {

        if inputs.rows() < self.k {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "inputs number of rows must be equal or learger than k"));
        }

        match (&self.tree, &self.target) {
            (&Some(ref tree), &Some(ref target)) => {

                let mut results: Vec<usize> = Vec::with_capacity(inputs.rows());
                for row in inputs.iter_rows() {
                    let (idx, _) = tree.search(row, self.k);
                    let res = target.select(&idx);
                    let (uniques, counts) = freq(&res.data());
                    let (id, _) = counts.argmax();
                    results.push(uniques[id]);
                }
                Ok(Vector::new(results))
            },
            _ => Err(Error::new_untrained()),
        }
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<usize>) -> LearningResult<()> {

        if inputs.rows() != targets.size() {
            return Err(Error::new(ErrorKind::InvalidData,
                                  "inputs and targets must be the same length"));
        }

        let mut tree = KDTree::new(inputs.clone(), self.leafsize);
        tree.build();

        self.tree = Some(tree);
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
        debug_assert!(index.len() == distances.len(), "index and distance must have the same length");
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

        // ToDo: use unsafe
        let len = self.pairs.len();

        let farthest = self.pairs[len - 1].1;
        if farthest < distance {
            if len < self.k {
                // append to the last
                self.pairs.push((index, distance));
                return distance;
            }
            return farthest;
        } else {
            // last element is already compared
            if len >= self.k {
                self.pairs.pop().unwrap();
            }
            for i in 2..(len + 1) {
                if self.pairs[len - i].1 < distance {
                    self.pairs.insert(len - i + 1, (index, distance));
                    if len < self.k {
                        return self.pairs[len].1;
                    } else {
                        return self.pairs[len - 1].1;
                    }
                }
            }
            self.pairs.insert(0, (index, distance));
            if len < self.k {
                return self.pairs[len].1;
            } else {
                return self.pairs[len - 1].1;
            }
        }
    }

    /// Return the k-th distance with searching point
    fn dist(&self) -> f64 {
        // KNearest should gather k element at least
        if self.pairs.len() < self.k {
            f64::MAX
        } else {
            self.pairs.last().unwrap().1
        }
    }

    /// Extract the search result to k-nearest indices and corresponding distances
    fn get_results(self) -> (Vec<usize>, Vec<f64>) {
        let mut indices: Vec<usize> = Vec::with_capacity(self.k);
        let mut distances: Vec<f64> = Vec::with_capacity(self.k);
        for (i, d) in self.pairs.into_iter() {
            indices.push(i);
            distances.push(d);
        }
        (indices, distances)
    }
}

/// Search K-nearest items
pub trait KNearestSearch {

    /// build data structure for search optimization
    fn build(&mut self) {
    }
    /// Serch k-nearest items close to the point
    /// Returns a tuple of searched item index and its distances
    fn search(&self, point: &[f64], k: usize) -> (Vec<usize>, Vec<f64>);
}

/// Count target label frequencies
/// ToDo: Used in decisition tree, move impl to somewhere
fn freq(labels: &[usize]) -> (Vector<usize>, Vector<usize>) {
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    for l in labels {
        let e = map.entry(*l).or_insert(0);
        *e += 1;
    }

    let mut uniques: Vec<usize> = Vec::with_capacity(map.len());
    let mut counts: Vec<usize> = Vec::with_capacity(map.len());
    for (&k, &v) in map.iter() {
        uniques.push(k);
        counts.push(v);
    }
    (Vector::new(uniques), Vector::new(counts))
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
