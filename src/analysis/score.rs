//! Functions for scoring a set of predictions, i.e. evaluating
//! how close predictions and truth are. All functions in this
//! module obey the convention that higher is better.

use linalg::{BaseMatrix, Matrix};
use learning::toolkit::cost_fn::{CostFunc, MeanSqError};

/// Returns the fraction of outputs which match their target.
pub fn accuracy<I>(outputs: I, targets: I) -> f64
    where I: ExactSizeIterator,
          I::Item: PartialEq
{
    assert!(outputs.len() == targets.len());
    let len = outputs.len() as f64;
    let correct = outputs
        .zip(targets)
        .filter(|&(ref x, ref y)| x == y)
        .count();
    correct as f64 / len
}


/// Returns the fraction of outputs rows which match their target.
pub fn row_accuracy<T: PartialEq>(outputs: &Matrix<T>, targets: &Matrix<T>) -> f64 {

    assert!(outputs.rows() == targets.rows());
    let len = outputs.rows() as f64;

    let correct = outputs.row_iter()
                         .zip(targets.row_iter())
                         .filter(|&(ref x, ref y)| x.raw_slice()
                                                    .iter()
                                                    .zip(y.raw_slice())
                                                    .all(|(v1, v2)| v1 == v2))
                         .count();
    correct as f64 / len

    // Row doesn't impl PartialEq
    // accuracy(outputs.row_iter(), targets.row_iter())
}

// TODO: generalise to accept arbitrary iterators of diff-able things
/// Returns the additive inverse of the mean-squared-error of the
/// outputs. So higher is better, and the returned value is always
/// negative.
pub fn neg_mean_squared_error(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64
{
    // MeanSqError divides the actual mean squared error by two.
    -2f64 * MeanSqError::cost(outputs, targets)
}

#[cfg(test)]
mod tests {
    use linalg::Matrix;
    use super::{accuracy, row_accuracy, neg_mean_squared_error};

    #[test]
    fn test_accuracy() {
        let outputs = [1, 2, 3, 4, 5, 6];
        let targets = [1, 2, 3, 3, 5, 1];
        assert_eq!(accuracy(outputs.iter(), targets.iter()), 2f64/3f64);
    }

    #[test]
    fn test_row_accuracy() {
        let outputs = matrix![1, 0;
                              0, 1;
                              1, 0];
        let targets = matrix![1, 0;
                              0, 1;
                              1, 0];
        assert_eq!(row_accuracy(&outputs, &targets), 1.0);

        let outputs = matrix![1, 0;
                              0, 1;
                              1, 0];
        let targets = matrix![0, 1;
                              0, 1;
                              1, 0];
        assert_eq!(row_accuracy(&outputs, &targets), 2. / 3.);

        let outputs = matrix![1., 0.;
                              0., 1.;
                              1., 0.];
        let targets = matrix![0., 1.;
                              0., 1.;
                              1., 0.];
        assert_eq!(row_accuracy(&outputs, &targets), 2. / 3.);
    }


    #[test]
    fn test_neg_mean_squared_error_1d() {
        let outputs = Matrix::new(3, 1, vec![1f64, 2f64, 3f64]);
        let targets = Matrix::new(3, 1, vec![2f64, 4f64, 3f64]);
        assert_eq!(neg_mean_squared_error(&outputs, &targets), -5f64/3f64);
    }

    #[test]
    fn test_neg_mean_squared_error_2d() {
        let outputs = Matrix::new(3, 2, vec![
            1f64, 2f64,
            3f64, 4f64,
            5f64, 6f64
            ]);
        let targets = Matrix::new(3, 2, vec![
            1.5f64, 2.5f64,
            5f64,   6f64,
            5.5f64, 6.5f64
            ]);
        assert_eq!(neg_mean_squared_error(&outputs, &targets), -3f64);
    }
}
