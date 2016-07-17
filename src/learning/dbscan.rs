//! DBSCAN

use learning::UnSupModel;

use linalg::{Matrix, Vector};
use rulinalg::utils;

/// DBSCAN Model
#[derive(Debug)]
pub struct DBSCAN {
    eps: f64,
    min_points: usize,
    clusters: Option<Vector<Option<usize>>>,
    _visited: Vec<bool>,
}

impl Default for DBSCAN {
    fn default() -> DBSCAN {
        DBSCAN {
            eps: 0.5,
            min_points: 5,
            clusters: None,
            _visited: Vec::new(),
        }
    }
}

impl UnSupModel<Matrix<f64>, Vector<usize>> for DBSCAN {
    /// Train the classifier using input data.
    fn train(&mut self, inputs: &Matrix<f64>) {
        self.init_params(inputs.rows());
        let mut cluster = 0;

        for (idx, point) in inputs.iter_rows().enumerate() {
            let visited = self._visited[idx];

            if !visited {
                self._visited[idx] = true;

                let neighbours = self.region_query(point, inputs);

                if neighbours.len() >= self.min_points {
                    self.expand_cluster(inputs, idx, neighbours, cluster);
                    cluster += 1;
                }
            }
        }
    }

    fn predict(&self, _: &Matrix<f64>) -> Vector<usize> {
        unimplemented!();
    }
}

impl DBSCAN {
    /// Create a new DBSCAN model.
    pub fn new(eps: f64, min_points: usize) -> DBSCAN {
        assert!(eps > 0f64, "The model epsilon must be positive.");

        DBSCAN {
            eps: eps,
            min_points: min_points,
            clusters: None,
            _visited: Vec::new(),
        }
    }

    /// Return an Option pointing to the model clusters.
    pub fn cluster(&self) -> Option<&Vector<Option<usize>>> {
        self.clusters.as_ref()
    }

    fn expand_cluster(&mut self,
                      inputs: &Matrix<f64>,
                      point_idx: usize,
                      neighbour_pts: Vec<usize>,
                      cluster: usize) {
        debug_assert!(point_idx < inputs.rows(),
                      "Point index too large for inputs");
        debug_assert!(neighbour_pts.iter().all(|x| *x < inputs.rows()),
                      "Neighbour indices too large for inputs");

        self.clusters.as_mut().map(|x| x.mut_data()[point_idx] = Some(cluster));

        for data_point_idx in &neighbour_pts {
            let visited = self._visited[*data_point_idx];
            if !visited {
                self._visited[*data_point_idx] = true;
                let sub_neighbours =
                    self.region_query(&inputs.data()[data_point_idx * inputs.cols()..(data_point_idx +
                                                                                   1) *
                                                                                  inputs.cols()],
                                      inputs);

                if sub_neighbours.len() >= self.min_points {
                    self.expand_cluster(inputs, *data_point_idx, sub_neighbours, cluster);
                }
            }
        }
    }


    fn region_query(&self, point: &[f64], inputs: &Matrix<f64>) -> Vec<usize> {
        debug_assert!(point.len() == inputs.cols(),
                      "point must be of same dimension as inputs");

        let mut in_neighbourhood = Vec::new();
        for (idx, data_point) in inputs.iter_rows().enumerate() {
            let point_distance = utils::vec_bin_op(data_point, point, |x, y| x - y);
            let dist = utils::dot(&point_distance, &point_distance).sqrt();

            if dist < self.eps {
                in_neighbourhood.push(idx);
            }
        }

        in_neighbourhood
    }

    fn init_params(&mut self, total_points: usize) {
        unsafe {
            self._visited.reserve(total_points);
            self._visited.set_len(total_points);
        }

        for i in 0..total_points {
            self._visited[i] = false;
        }

        self.clusters = Some(Vector::new(vec![None; total_points]));
    }
}

#[cfg(test)]
mod tests {
    use super::DBSCAN;
    use linalg::Matrix;

    #[test]
    fn test_region_query() {
        let model = DBSCAN::new(1.0, 3);

        let inputs = Matrix::new(3, 2, vec![1.0, 1.0, 1.1, 1.9, 3.0, 3.0]);

        let neighbours = model.region_query(&[1.0, 1.0], &inputs);

        assert!(neighbours.len() == 2);
    }

    #[test]
    fn test_region_query_small_eps() {
        let model = DBSCAN::new(0.01, 3);

        let inputs = Matrix::new(3, 2, vec![1.0, 1.0, 1.1, 1.9, 1.1, 1.1]);

        let neighbours = model.region_query(&[1.0, 1.0], &inputs);

        assert!(neighbours.len() == 1);
    }
}
