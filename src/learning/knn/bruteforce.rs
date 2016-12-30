//! Bruteforce search implementations
use linalg::{Matrix, BaseMatrix};

use super::{KNearest, KNearestSearch};

struct BruteForce {
    data: Matrix<f64>,
}

impl BruteForce {
    /// initialize KDTree, must call .build to actually built tree
    pub fn new(data: Matrix<f64>) -> Self {
        BruteForce {
            data: data
        }
    }

    /// return distances between given point and data specified with row ids
    fn get_distances(&self, point: &[f64], ids: &[usize]) -> Vec<f64> {
        // ToDo: merge impl as KDTree
        assert!(ids.len() > 0, "target ids is empty");

        let mut distances: Vec<f64> = Vec::with_capacity(ids.len());
        for id in ids.iter() {
            // ToDo: use .row(*id)
            let row: Vec<f64> = self.data.select_rows(&[*id]).into_vec();
            // let row: Vec<f64> = self.data.row(*id).into_vec();
            let d = dist(point, &row);
            distances.push(d);
        }
        distances
    }
}

/// Can search K-nearest items
impl KNearestSearch for BruteForce {
    /// Serch k-nearest items close to the point
    fn search(&self, point: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
        let indices: Vec<usize> = (0..k).collect();
        let distances = self.get_distances(point, &indices);

        let mut query = KNearest::new(k, indices, distances);
        let mut current_dist = query.dist();

        let mut i = k;
        for row in self.data.iter_rows().skip(k) {
            // ToDo: Do not instanciate Vec
            let row: Vec<f64> = row.iter().cloned().collect();
            let d = dist(point, &row);
            // ToDo: rewrite to add single elements
            if d < current_dist {
                current_dist = query.add(i, d);
            }
            i += 1;
        }
        query.get_results()
    }
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

    use linalg::Matrix;
    use super::super::KNearestSearch;
    use super::BruteForce;

    #[test]
    fn test_bruteforce_search() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut b = BruteForce::new(m);
        b.build();  // no op

        let (ind, dist) = b.search(&vec![3., 4.9], 1);
        assert_eq!(ind, vec![3]);
        assert_eq!(dist, vec![1.0999999999999996]);

        let (ind, dist) = b.search(&vec![3., 4.9], 2);
        assert_eq!(ind, vec![3, 0]);
        assert_eq!(dist, vec![1.0999999999999996, 3.5227829907617076]);

        let (ind, dist) = b.search(&vec![3., 4.9], 3);
        assert_eq!(ind, vec![3, 0, 4]);
        assert_eq!(dist, vec![1.0999999999999996, 3.5227829907617076, 3.551056180912941]);
    }
}
