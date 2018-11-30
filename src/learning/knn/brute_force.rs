//! Bruteforce search implementations
use learning::error::Error;
use linalg::{BaseMatrix, Matrix};

use super::{dist, get_distances, KNearest, KNearestSearch};

/// Perform brute-force search
#[derive(Debug)]
pub struct BruteForce {
    data: Option<Matrix<f64>>,
}

impl Default for BruteForce {
    /// Constructs new brute-force search
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::BruteForce;
    /// let _ = BruteForce::default();
    /// ```
    fn default() -> Self {
        BruteForce { data: None }
    }
}

impl BruteForce {
    /// Constructs new brute-force search.
    /// BruteForce accepts no parapeters.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::knn::BruteForce;
    /// let _ = BruteForce::new();
    /// ```
    pub fn new() -> Self {
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
    fn search(&self, point: &[f64], k: usize) -> Result<(Vec<usize>, Vec<f64>), Error> {
        if let Some(ref data) = self.data {
            let indices: Vec<usize> = (0..k).collect();
            let distances = get_distances(data, point, &indices);

            let mut query = KNearest::new(k, indices, distances);
            let mut current_dist = query.dist();

            let mut i = k;
            for row in data.row_iter().skip(k) {
                let d = dist(point, row.raw_slice());
                if d < current_dist {
                    current_dist = query.add(i, d);
                }
                i += 1;
            }
            Ok(query.get_results())
        } else {
            Err(Error::new_untrained())
        }
    }
}

#[cfg(test)]
mod tests {

    use super::super::KNearestSearch;
    use super::BruteForce;
    use linalg::Matrix;

    #[test]
    fn test_bruteforce_search() {
        let m = Matrix::new(5, 2, vec![1., 2., 8., 0., 6., 10., 3., 6., 0., 3.]);
        let mut b = BruteForce::new();
        b.build(m);

        let (ind, dist) = b.search(&vec![3., 4.9], 1).unwrap();
        assert_eq!(ind, vec![3]);
        assert_eq!(dist, vec![1.0999999999999996]);

        let (ind, dist) = b.search(&vec![3., 4.9], 2).unwrap();
        assert_eq!(ind, vec![3, 0]);
        assert_eq!(dist, vec![1.0999999999999996, 3.5227829907617076]);

        let (ind, dist) = b.search(&vec![3., 4.9], 3).unwrap();
        assert_eq!(ind, vec![3, 0, 4]);
        assert_eq!(
            dist,
            vec![1.0999999999999996, 3.5227829907617076, 3.551056180912941]
        );
    }

    #[test]
    fn test_bruteforce_untrained() {
        let b = BruteForce::new();
        let e = b.search(&vec![3., 4.9], 1);
        assert!(e.is_err());
    }
}
