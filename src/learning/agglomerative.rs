//! Agglomerative (Hierarchical) Clustering Module
//!
//! Contains implementation of Agglomerative Clustering.
//!
//! # Usage
//!
//! ```
//! use rusty_machine::learning::agglomerative::{AgglomerativeClustering, Linkage};
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::{Matrix, Vector};
//!
//! let inputs = Matrix::new(4, 2, vec![1., 3., 2., 3., 4., 3., 5., 3.]);
//! let mut agg = AgglomerativeClustering::new(2, Linkage::Single);
//!
//! // Train the model and get the clustering result
//! let res = agg.train(&inputs).unwrap();
//!
//! assert_eq!(res, Vector::new(vec![0, 0, 1, 1]));
//! ```
//! ```

use std::collections::{BTreeMap, HashMap};
use std::f64;

use linalg::{Matrix, BaseMatrix, Vector};
use learning::{LearningResult};

/// Agglomerative clustering distances
#[derive(Debug)]
pub enum Linkage {
    /// Single linkage clustering
    Single,
    /// Complete linkage clustering
    Complete,
    /// Average linkage clustering
    Average,
    /// Centroid linkage clustering
    Centroid,
    /// Median linkage clustering
    Median,

    /// Ward criterion (uses Ward II)
    Ward,
    /// Ward I,
    /// See "Ward’s Hierarchical Agglomerative Clustering Method:
    /// Which Algorithms Implement Ward’s Criterion? (Murtagh, 2014)"
    Ward1,
    /// Ward II
    Ward2,
}

impl Linkage {

    // calculate distance using Lance-Williams algorithm
    fn dist(&self, ci: &Cluster, cj: &Cluster, ck: &Cluster, dmat: &DistanceMatrix) -> f64 {

        let dik = dmat.get(ck.id, ci.id);
        let djk = dmat.get(ck.id, cj.id);

        match self {
            &Linkage::Single => {
                // 0.5 * dik + 0.5 * djk + 0. * dij - 0.5 * (dik - djk).abs()
                dik.min(djk)
            },
            &Linkage::Complete =>  {
                // 0.5 * dik + 0.5 * djk + 0. * dij + 0.5 * (dik - djk).abs()
                dik.max(djk)
            },
            &Linkage::Average =>  {
                let s = ci.size + cj.size;
                ci.size / s * dik + cj.size / s * djk
            },
            &Linkage::Centroid => {
                let s = ci.size + cj.size;
                let ai = ci.size / s;
                let aj = cj.size / s;
                let dij = dmat.get(ci.id, cj.id);
                ai * dik + aj * djk - ai * aj * dij
            },
            &Linkage::Median => {
                let dij = dmat.get(ci.id, cj.id);
                0.5 * dik + 0.5 * djk - 0.25 * dij
            },
            &Linkage::Ward1  => {
                let s = ci.size + cj.size + ck.size;
                let dij = dmat.get(ci.id, cj.id);
                (ci.size + ck.size) / s * dik + (cj.size + ck.size) / s * djk - ck.size / s * dij
            },
            &Linkage::Ward | &Linkage::Ward2 => {
                let s = ci.size + cj.size + ck.size;
                let dij = dmat.get(ci.id, cj.id);
                ((ci.size + ck.size) / s * dik * dik + (cj.size + ck.size) / s * djk * djk - ck.size / s * dij * dij).sqrt()
            }
        }
    }
}

struct Cluster {
    /// Cluster id
    id: usize,
    /// Number of nodes (rows) which belongs to cluster
    /// to avoid cast in the algorithm, store it as f64
    size: f64,
    /// Row ids belong to the cluster
    nodes: Vec<usize>,
}

impl Cluster {

    /// Create new cluster
    fn new(id: usize, nodes: Vec<usize>) -> Cluster {
        Cluster {
            id: id,
            size: 1.,
            nodes: nodes,
        }
    }

    /// Create new cluster merging left and right
    fn from_clusters(id: usize, left: Cluster, mut right: Cluster) -> Cluster {
        let mut new_nodes = left.nodes;
        new_nodes.append(&mut right.nodes);
        Cluster {
            id: id,
            size: left.size + right.size,
            nodes: new_nodes
        }
    }
}

/// Distance Matrix
#[derive(Debug)]
struct DistanceMatrix {
    // Distance is symmetric, no need to hold all pairs
    // use HashMap to easier update
    data: HashMap<(usize, usize), f64>
}

impl DistanceMatrix {

    /// Create distance matrix fron input matrix
    fn from_mat(inputs: &Matrix<f64>) -> Self {
        assert!(inputs.rows() > 0, "input is empty");

        let n = inputs.rows() - 1;
        let mut data: HashMap<(usize, usize), f64> = HashMap::with_capacity(n * n);

        unsafe {
            for i in 0..n {
                for j in (i + 1)..inputs.rows() {
                    let mut val = 0.;
                    for k in 0..inputs.cols() {
                        let d = inputs.get_unchecked([i, k]) - inputs.get_unchecked([j, k]);
                        val += d * d;
                    }
                    val = val.sqrt();
                    data.insert((i, j), val);
                }
            }
        }
        DistanceMatrix {
            data: data
        }
    }

    /// Get distance between i-th and j-th item
    fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            0.
        } else if i > j {
            *self.data.get(&(j, i)).unwrap()
        } else {
            *self.data.get(&(i, j)).unwrap()
        }
    }

    /// Add distance between i-th and j-th item
    /// i must be smaller than j
    fn insert(&mut self, i: usize, j: usize, dist: f64) {
        debug_assert!(i < j, "i must be smaller than j");
        self.data.insert((i, j), dist);
    }

    /// Delete distance between i-th and j-th item
    fn delete(&mut self, i: usize, j: usize) {
        debug_assert!(i != j, "DistanceMatrix doesn't store distance when i == j, because it is 0.0");
        if i > j {
            self.data.remove(&(j, i));
        } else {
            self.data.remove(&(i, j));
        }
    }
}

/// Agglomerative clustering
#[derive(Debug)]
pub struct AgglomerativeClustering {
    n: usize,
    linkage: Linkage,

    // internally stores distances / merged history (currently for testing)
    distances: Option<Vec<f64>>,
    merged: Option<Vec<(usize, usize)>>
}

impl AgglomerativeClustering {

    /// Constructs an untrained Decision Tree with specified
    ///
    /// - `n` - Number of clusters
    /// - `linkage` - Linkage method
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::agglomerative::{AgglomerativeClustering, Linkage};
    ///
    /// let _ = AgglomerativeClustering::new(3, Linkage::Single);
    /// ```
    pub fn new(n: usize, linkage: Linkage) -> Self {
        AgglomerativeClustering {
            n: n,
            linkage: linkage,

            distances: None,
            merged: None
        }
    }

    /// train the data and predict the cluster
    pub fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<Vector<usize>> {
        let mut dmat = DistanceMatrix::from_mat(&inputs);

        // initialize cluster
        let mut clusters: Vec<Cluster> = (0..inputs.rows()).map(|i| Cluster::new(i, vec![i]))
                                                           .collect();;
        // vec to store merged cluster distances
        let mut distances: Vec<f64> = Vec::with_capacity(inputs.rows() - self.n);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(inputs.rows() - self.n);

        let mut id = inputs.rows();
        while clusters.len() > self.n {
            let mut tmp_i = 0;
            let mut tmp_j = 0;
            let mut current_dist = f64::MAX;

            // loop with index to remember the position to be removed
            for i in 0..(clusters.len() - 1) {
                for j in (i + 1)..clusters.len() {
                    let ci = unsafe { clusters.get_unchecked(i) };
                    let cj = unsafe { clusters.get_unchecked(j) };

                    let d = dmat.get(ci.id, cj.id);
                    if d < current_dist {
                        current_dist = d;
                        tmp_i = i;
                        tmp_j = j;
                    }
                }
            }

            distances.push(current_dist);

            // update cluster
            // cj must be first because j > i
            let cj = clusters.swap_remove(tmp_j);
            let ci = clusters.swap_remove(tmp_i);
            merged.push((ci.id, cj.id));

            // update distances using Lance Williams algorithm
            for ck in clusters.iter() {
                let d = self.linkage.dist(&ci, &cj, ck, &dmat);
                dmat.insert(ck.id, id, d);

                // remove unnecessary distances
                dmat.delete(ck.id, ci.id);
                dmat.delete(ck.id, cj.id);
            }

            let new = Cluster::from_clusters(id, ci, cj);
            id += 1;
            clusters.push(new);
        }
        // store distances
        self.distances = Some(distances);
        // store merged history
        self.merged = Some(merged);

        let mut sorter: BTreeMap<usize, usize> = BTreeMap::new();
        for (i, c) in clusters.iter().enumerate() {
            for n in c.nodes.iter() {
                sorter.insert(*n, i);
            }
        }
        let res: Vec<usize> = sorter.values().cloned().collect();
        Ok(Vector::new(res))
    }
}


#[cfg(test)]
mod tests {

    use super::{AgglomerativeClustering, DistanceMatrix, Linkage};

    #[test]
    fn test_distance_matrix() {
        let data = matrix![1., 2.;
                           2., 3.;
                           0., 5.;
                           3., 3.];

        let m = DistanceMatrix::from_mat(&data);

        assert_eq!(m.get(0, 0), 0.);

        assert_eq!(m.get(0, 1), 2.0f64.sqrt());
        assert_eq!(m.get(1, 0), 2.0f64.sqrt());

        assert_eq!(m.get(0, 2), 10.0f64.sqrt());
        assert_eq!(m.get(2, 0), 10.0f64.sqrt());

        assert_eq!(m.get(0, 3), 5.0f64.sqrt());
        assert_eq!(m.get(3, 0), 5.0f64.sqrt());

        assert_eq!(m.get(1, 1), 0.);

        assert_eq!(m.get(1, 2), 8.0f64.sqrt());
        assert_eq!(m.get(2, 1), 8.0f64.sqrt());

        assert_eq!(m.get(1, 3), 1.);
        assert_eq!(m.get(3, 1), 1.);

        assert_eq!(m.get(2, 2), 0.);

        assert_eq!(m.get(2, 3), 13.0f64.sqrt());
        assert_eq!(m.get(3, 2), 13.0f64.sqrt());
    }

    #[test]
    fn test_distances() {
        // test distances are calculated propery
        let data = matrix![89., 90., 67. ,46., 50.;
                           57., 70., 80., 85., 90.;
                           80., 90., 35., 40., 50.;
                           40., 60., 50., 45., 55.;
                           78., 85., 45., 55., 60.;
                           55., 65., 80., 75., 85.;
                           90., 85., 88., 92., 95.];

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Single);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 28.478061731796284,
                       38.1051177665153, 47.10626285325551, 54.31390245600108];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Complete);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 33.77869150810907,
                       45.58508528016593, 60.13318551349163, 91.53141537199127];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Average);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 31.128376619952675,
                       41.84510152334062, 53.305905710336944, 69.92295649225116];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Centroid);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 25.801557681787045,
                       38.7426831118429, 44.021013600051624, 44.02758328256392];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Median);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 25.801557681787045,
                       38.7426831118429, 45.898926771596045, 45.42216730738696];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Ward1);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 34.4020769090494,
                       51.65691081579053, 66.03152040007744, 150.95171411164773];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Ward2);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 33.911649915626334,
                       47.97916214358062, 62.48199740725323, 115.91869071527186];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);

        let mut hclust = AgglomerativeClustering::new(1, Linkage::Ward);
        let _ = hclust.train(&data);
        let exp = vec![12.409673645990857, 21.307275752662516, 33.911649915626334,
                       47.97916214358062, 62.48199740725323, 115.91869071527186];
        assert_eq!(hclust.distances.unwrap(), exp);
        let exp = vec![(1, 5), (2, 4), (0, 8), (6, 7), (3, 9), (10, 11)];
        assert_eq!(hclust.merged.unwrap(), exp);
    }
}
