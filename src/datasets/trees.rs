use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

use super::Dataset;

/// Load trees dataset.
///
/// The data set contains a sample of 31 black cherry trees in the
/// Allegheny National Forest, Pennsylvania.
///
/// ## Attribute Information
///
/// ### Data
///
/// ``Matrix<f64>`` contains following columns.
///
///   - diameter (inches)
///   - height (feet)
///
/// ### Target
///
/// ``Vector<f64>`` contains volume (cubic feet) of trees.
///
/// Thomas A. Ryan, Brian L. Joiner, Barbara F. Ryan. (1976).
/// Minitab student handbook. Duxbury Press
pub fn load() -> Dataset<Matrix<f64>, Vector<f64>> {
    let data = matrix![8.3, 70.;
                       8.6, 65.;
                       8.8, 63.;
                       10.5, 72.;
                       10.7, 81.;
                       10.8, 83.;
                       11.0, 66.;
                       11.0, 75.;
                       11.1, 80.;
                       11.2, 75.;
                       11.3, 79.;
                       11.4, 76.;
                       11.4, 76.;
                       11.7, 69.;
                       12.0, 75.;
                       12.9, 74.;
                       12.9, 85.;
                       13.3, 86.;
                       13.7, 71.;
                       13.8, 64.;
                       14.0, 78.;
                       14.2, 80.;
                       14.5, 74.;
                       16.0, 72.;
                       16.3, 77.;
                       17.3, 81.;
                       17.5, 82.;
                       17.9, 80.;
                       18.0, 80.;
                       18.0, 80.;
                       20.6, 87.];
    let target = vec![10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9,
                      24.2, 21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9,
                      34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0,
                      77.0];
    Dataset{ data: data,
             target: Vector::new(target) }
}