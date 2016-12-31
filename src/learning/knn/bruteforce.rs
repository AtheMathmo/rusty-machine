//! Bruteforce search implementations
use linalg::{Matrix, BaseMatrix};

use super::{KNearest, KNearestSearch, get_distances, dist};

/// Perform brute-force search
#[derive(Debug)]
pub struct BruteForce {
    data: Option<Matrix<f64>>,
}

impl Default for BruteForce {
    fn default() -> Self {
        BruteForce {
            data: None
        }
    }
}

impl BruteForce {
    fn new() -> Self {
        BruteForce::default()
    }
}

/// Can search K-nearest items
impl KNearestSearch for BruteForce {

    /// initialize BruteForce Searcher
    fn build(&mut self, data: Matrix<f64>) {
        self.data = Some(data);
    }

    /// Serch k-nearest items close to the point
    fn search(&self, point: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
        if let &Some(ref data) = &self.data {
            let indices: Vec<usize> = (0..k).collect();
            let distances = get_distances(data, point, &indices);

            let mut query = KNearest::new(k, indices, distances);
            let mut current_dist = query.dist();

            let mut i = k;
            for row in data.iter_rows().skip(k) {
                let d = dist(point, &row);
                if d < current_dist {
                    current_dist = query.add(i, d);
                }
                i += 1;
            }
            query.get_results()
        } else {
            panic!("error")
        }
    }
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
        let mut b = BruteForce::new();
        b.build(m);

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
