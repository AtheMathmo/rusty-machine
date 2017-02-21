//! Functions for scoring a set of predictions, i.e. evaluating
//! how close predictions and truth are. All functions in this
//! module obey the convention that higher is better.

use libnum::{Zero, One};

use linalg::{BaseMatrix, Matrix};
use learning::toolkit::cost_fn::{CostFunc, MeanSqError};

// ************************************
// Classification Scores
// ************************************

/// Returns the fraction of outputs which match their target.
///
/// # Arguments
///
/// * `outputs` - Iterator of output (predicted) labels.
/// * `targets` - Iterator of expected (actual) labels.
///
/// # Examples
///
/// ```
/// use rusty_machine::analysis::score::accuracy;
/// let outputs = [1, 1, 1, 0, 0, 0];
/// let targets = [1, 1, 0, 0, 1, 1];
///
/// assert_eq!(accuracy(outputs.iter(), targets.iter()), 0.5);
/// ```
///
/// # Panics
///
/// - outputs and targets have different length
pub fn accuracy<I1, I2, T>(outputs: I1, targets: I2) -> f64
    where T: PartialEq,
          I1: ExactSizeIterator + Iterator<Item=T>,
          I2: ExactSizeIterator + Iterator<Item=T>
{
    assert!(outputs.len() == targets.len(), "outputs and targets must have the same length");
    let len = outputs.len() as f64;
    let correct = outputs
        .zip(targets)
        .filter(|&(ref x, ref y)| x == y)
        .count();
    correct as f64 / len
}

/// Returns the fraction of outputs rows which match their target.
pub fn row_accuracy(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
    accuracy(outputs.row_iter().map(|r| r.raw_slice()),
             targets.row_iter().map(|r| r.raw_slice()))
}

/// Returns the precision score for 2 class classification.
///
/// Precision is calculated with true-positive / (true-positive + false-positive),
/// see [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) for details.
///
/// # Arguments
///
/// * `outputs` - Iterator of output (predicted) labels which only contains 0 or 1.
/// * `targets` - Iterator of expected (actual) labels which only contains 0 or 1.
///
/// # Examples
///
/// ```
/// use rusty_machine::analysis::score::precision;
/// let outputs = [1, 1, 1, 0, 0, 0];
/// let targets = [1, 1, 0, 0, 1, 1];
///
/// assert_eq!(precision(outputs.iter(), targets.iter()), 2.0f64 / 3.0f64);
/// ```
///
/// # Panics
///
/// - outputs and targets have different length
/// - outputs or targets contains a value which is not 0 or 1
pub fn precision<'a, I, T>(outputs: I, targets: I) -> f64
    where I: ExactSizeIterator<Item=&'a T>,
          T: 'a + PartialEq + Zero + One
{
    assert!(outputs.len() == targets.len(), "outputs and targets must have the same length");

    let mut tpfp = 0.0f64;
    let mut tp = 0.0f64;

    for (ref o, ref t) in outputs.zip(targets) {
        if *o == &T::one() {
            tpfp += 1.0f64;
            if *t == &T::one() {
                tp += 1.0f64;
            }
        }
        if ((*t != &T::zero()) & (*t != &T::one())) |
           ((*o != &T::zero()) & (*o != &T::one())) {
            panic!("precision must be used for 2 class classification")
        }
    }
    tp / tpfp
}

/// Returns the recall score for 2 class classification.
///
/// Recall is calculated with true-positive / (true-positive + false-negative),
/// see [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) for details.
///
/// # Arguments
///
/// * `outputs` - Iterator of output (predicted) labels which only contains 0 or 1.
/// * `targets` - Iterator of expected (actual) labels which only contains 0 or 1.
///
/// # Examples
///
/// ```
/// use rusty_machine::analysis::score::recall;
/// let outputs = [1, 1, 1, 0, 0, 0];
/// let targets = [1, 1, 0, 0, 1, 1];
///
/// assert_eq!(recall(outputs.iter(), targets.iter()), 0.5);
/// ```
///
/// # Panics
///
/// - outputs and targets have different length
/// - outputs or targets contains a value which is not 0 or 1
pub fn recall<'a, I, T>(outputs: I, targets: I) -> f64
    where I: ExactSizeIterator<Item=&'a T>,
          T: 'a + PartialEq + Zero + One
{
    assert!(outputs.len() == targets.len(), "outputs and targets must have the same length");

    let mut tpfn = 0.0f64;
    let mut tp = 0.0f64;

    for (ref o, ref t) in outputs.zip(targets) {
        if *t == &T::one() {
            tpfn += 1.0f64;
            if *o == &T::one() {
                tp += 1.0f64;
            }
        }
        if ((*t != &T::zero()) & (*t != &T::one())) |
           ((*o != &T::zero()) & (*o != &T::one())) {
            panic!("recall must be used for 2 class classification")
        }
    }
    tp / tpfn
}

/// Returns the f1 score for 2 class classification.
///
/// F1-score is calculated with 2 * precision * recall / (precision + recall),
/// see [F1 score](https://en.wikipedia.org/wiki/F1_score) for details.
///
/// # Arguments
///
/// * `outputs` - Iterator of output (predicted) labels which only contains 0 or 1.
/// * `targets` - Iterator of expected (actual) labels which only contains 0 or 1.
///
/// # Examples
///
/// ```
/// use rusty_machine::analysis::score::f1;
/// let outputs = [1, 1, 1, 0, 0, 0];
/// let targets = [1, 1, 0, 0, 1, 1];
///
/// assert_eq!(f1(outputs.iter(), targets.iter()), 0.5714285714285714);
/// ```
///
/// # Panics
///
/// - outputs and targets have different length
/// - outputs or targets contains a value which is not 0 or 1
pub fn f1<'a, I, T>(outputs: I, targets: I) -> f64
    where I: ExactSizeIterator<Item=&'a T>,
          T: 'a + PartialEq + Zero + One
{
    assert!(outputs.len() == targets.len(), "outputs and targets must have the same length");

    let mut tpos = 0.0f64;
    let mut fpos = 0.0f64;
    let mut fneg = 0.0f64;

    for (ref o, ref t) in outputs.zip(targets) {
        if (*o == &T::one()) & (*t == &T::one()) {
            tpos += 1.0f64;
        } else if *t == &T::one() {
            fpos += 1.0f64;
        } else if *o == &T::one() {
            fneg += 1.0f64;
        }
        if ((*t != &T::zero()) & (*t != &T::one())) |
           ((*o != &T::zero()) & (*o != &T::one())) {
            panic!("f1-score must be used for 2 class classification")
        }
    }
    2.0f64 * tpos / (2.0f64 * tpos + fneg + fpos)
}

// ************************************
// Regression Scores
// ************************************

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
    use super::{accuracy, precision, recall, f1, neg_mean_squared_error};

    #[test]
    fn test_accuracy() {
        let outputs = [1, 2, 3, 4, 5, 6];
        let targets = [1, 2, 3, 3, 5, 1];
        assert_eq!(accuracy(outputs.iter(), targets.iter()), 2f64/3f64);

        let outputs = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 1, 0, 0, 1];
        assert_eq!(accuracy(outputs.iter(), targets.iter()), 5.0f64 / 6.0f64);
    }

    #[test]
    fn test_precision() {
        let outputs = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(precision(outputs.iter(), targets.iter()), 2.0f64 / 3.0f64);

        let outputs = [1, 1, 1, 0, 1, 1];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(precision(outputs.iter(), targets.iter()), 0.8);

        let outputs = [0, 0, 0, 1, 1, 1];
        let targets = [1, 1, 1, 1, 1, 0];
        assert_eq!(precision(outputs.iter(), targets.iter()), 2.0f64 / 3.0f64);

        let outputs = [1, 1, 1, 1, 1, 0];
        let targets = [0, 0, 0, 1, 1, 1];
        assert_eq!(precision(outputs.iter(), targets.iter()), 0.4);
    }

    #[test]
    #[should_panic]
    fn test_precision_outputs_not_2class() {
        let outputs = [1, 2, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        precision(outputs.iter(), targets.iter());
    }

    #[test]
    #[should_panic]
    fn test_precision_targets_not_2class() {
        let outputs = [1, 0, 1, 0, 0, 0];
        let targets = [1, 2, 0, 0, 1, 1];
        precision(outputs.iter(), targets.iter());
    }

    #[test]
    fn test_recall() {
        let outputs = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(recall(outputs.iter(), targets.iter()), 0.5);

        let outputs = [1, 1, 1, 0, 1, 1];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(recall(outputs.iter(), targets.iter()), 1.0);

        let outputs = [0, 0, 0, 1, 1, 1];
        let targets = [1, 1, 1, 1, 1, 0];
        assert_eq!(recall(outputs.iter(), targets.iter()), 0.4);

        let outputs = [1, 1, 1, 1, 1, 0];
        let targets = [0, 0, 0, 1, 1, 1];
        assert_eq!(recall(outputs.iter(), targets.iter()), 2.0f64 / 3.0f64);
    }

    #[test]
    #[should_panic]
    fn test_recall_outputs_not_2class() {
        let outputs = [1, 2, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        recall(outputs.iter(), targets.iter());
    }

    #[test]
    #[should_panic]
    fn test_recall_targets_not_2class() {
        let outputs = [1, 0, 1, 0, 0, 0];
        let targets = [1, 2, 0, 0, 1, 1];
        recall(outputs.iter(), targets.iter());
    }

    #[test]
    fn test_f1() {
        let outputs = [1, 1, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(f1(outputs.iter(), targets.iter()), 0.5714285714285714);

        let outputs = [1, 1, 1, 0, 1, 1];
        let targets = [1, 1, 0, 0, 1, 1];
        assert_eq!(f1(outputs.iter(), targets.iter()), 0.8888888888888888);

        let outputs = [0, 0, 0, 1, 1, 1];
        let targets = [1, 1, 1, 1, 1, 0];
        assert_eq!(f1(outputs.iter(), targets.iter()), 0.5);

        let outputs = [1, 1, 1, 1, 1, 0];
        let targets = [0, 0, 0, 1, 1, 1];
        assert_eq!(f1(outputs.iter(), targets.iter()), 0.5);
    }


    #[test]
    #[should_panic]
    fn test_f1_outputs_not_2class() {
        let outputs = [1, 2, 1, 0, 0, 0];
        let targets = [1, 1, 0, 0, 1, 1];
        f1(outputs.iter(), targets.iter());
    }

    #[test]
    #[should_panic]
    fn test_f1_targets_not_2class() {
        let outputs = [1, 0, 1, 0, 0, 0];
        let targets = [1, 2, 0, 0, 1, 1];
        f1(outputs.iter(), targets.iter());
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
