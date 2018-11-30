//! Module to compute the confusion matrix of a set of predictions.

use linalg::Matrix;
use std::collections::HashMap;
use std::hash::Hash;

/// Returns a square matrix C where C_ij is the count of the samples which were
/// predicted to lie in the class with jth label but actually lie in the class with
/// ith label.
///
/// # Arguments
/// * `predictions` - A series of model predictions.
/// * `targets`     - A slice of equal length to predictions, containing the
///                   target results.
/// * `labels`      - If None then the rows and columns of the returned matrix
///                   correspond to the distinct labels appearing in either
///                   predictions or targets, in increasing order.
///                   If Some then the rows and columns correspond to the provided
///                   labels, in the provided order. Note that in this case the
///                   confusion matrix will only contain entries for the elements
///                   of `labels`.
///
/// # Examples
/// ```
/// use rusty_machine::analysis::confusion_matrix::confusion_matrix;
/// use rusty_machine::linalg::Matrix;
///
/// let truth       = vec![2, 0, 2, 2, 0, 1];
/// let predictions = vec![0, 0, 2, 2, 0, 2];
///
/// let confusion = confusion_matrix(&predictions, &truth, None);
///
/// let expected = Matrix::new(3, 3, vec![
///     2, 0, 0,
///     0, 0, 1,
///     1, 0, 2]);
///
/// assert_eq!(confusion, expected);
/// ```
/// # Panics
///
/// - If user-provided labels are not distinct.
/// - If predictions and targets have different lengths.
pub fn confusion_matrix<T>(
    predictions: &[T],
    targets: &[T],
    labels: Option<Vec<T>>,
) -> Matrix<usize>
where
    T: Ord + Eq + Hash + Copy,
{
    assert!(
        predictions.len() == targets.len(),
        "predictions and targets have different lengths"
    );

    let labels = match labels {
        Some(ls) => ls,
        None => ordered_distinct(predictions, targets),
    };

    let mut label_to_index: HashMap<T, usize> = HashMap::new();
    for (i, l) in labels.iter().enumerate() {
        match label_to_index.insert(*l, i) {
            None => {}
            Some(_) => {
                panic!("labels must be distinct");
            }
        }
    }

    let mut counts = Matrix::new(
        labels.len(),
        labels.len(),
        vec![0usize; labels.len() * labels.len()],
    );

    for (truth, pred) in targets.iter().zip(predictions) {
        if label_to_index.contains_key(truth) && label_to_index.contains_key(pred) {
            let row = label_to_index[truth];
            let col = label_to_index[pred];

            counts[[row, col]] += 1;
        }
    }

    counts
}

fn ordered_distinct<T: Ord + Eq + Copy>(xs: &[T], ys: &[T]) -> Vec<T> {
    let mut ds: Vec<T> = xs.iter().chain(ys).cloned().collect();
    ds.sort();
    ds.dedup();
    ds
}

#[cfg(test)]
mod tests {
    use super::confusion_matrix;

    #[test]
    fn confusion_matrix_no_labels() {
        let truth = vec![2, 0, 2, 2, 0, 1];
        let predictions = vec![0, 0, 2, 2, 0, 2];

        let confusion = confusion_matrix(&predictions, &truth, None);

        let expected = matrix!(2, 0, 0;
                               0, 0, 1;
                               1, 0, 2);

        assert_eq!(confusion, expected);
    }

    #[test]
    fn confusion_matrix_with_labels_a_permutation_of_classes() {
        let truth = vec![2, 0, 2, 2, 0, 1];
        let predictions = vec![0, 0, 2, 2, 0, 2];

        let labels = vec![2, 1, 0];
        let confusion = confusion_matrix(&predictions, &truth, Some(labels));

        let expected = matrix!(2, 0, 1;
                               1, 0, 0;
                               0, 0, 2);

        assert_eq!(confusion, expected);
    }

    #[test]
    fn confusion_matrix_accepts_labels_intersecting_targets_and_disjoint_from_predictions() {
        let truth = vec![2, 0, 2, 2, 3, 1];
        let predictions = vec![0, 0, 2, 2, 0, 2];

        let labels = vec![1, 3];
        let confusion = confusion_matrix(&predictions, &truth, Some(labels));

        let expected = matrix!(0, 0;
                               0, 0);

        assert_eq!(confusion, expected);
    }

    #[test]
    fn confusion_matrix_accepts_labels_intersecting_predictions_and_disjoint_from_targets() {
        let truth = vec![0, 0, 2, 2, 0, 2];
        let predictions = vec![2, 0, 2, 2, 3, 1];

        let labels = vec![1, 3];
        let confusion = confusion_matrix(&predictions, &truth, Some(labels));

        let expected = matrix!(0, 0;
                               0, 0);

        assert_eq!(confusion, expected);
    }

    #[test]
    fn confusion_matrix_accepts_labels_disjoint_from_predictions_and_targets() {
        let truth = vec![0, 0, 2, 2, 0, 2];
        let predictions = vec![2, 0, 2, 2, 3, 1];

        let labels = vec![4, 5];
        let confusion = confusion_matrix(&predictions, &truth, Some(labels));

        let expected = matrix!(0, 0;
                               0, 0);

        assert_eq!(confusion, expected);
    }

    #[test]
    #[should_panic]
    fn confusion_matrix_rejects_duplicate_labels() {
        let truth = vec![0, 0, 2, 2, 0, 2];
        let predictions = vec![2, 0, 2, 2, 3, 1];

        let labels = vec![1, 1];
        let _ = confusion_matrix(&predictions, &truth, Some(labels));
    }

    #[test]
    #[should_panic]
    fn confusion_matrix_rejects_mismatched_prediction_and_target_lengths() {
        let truth = vec![0, 0, 2, 2, 0, 2];
        let predictions = vec![2, 0, 2, 2];
        let _ = confusion_matrix(&predictions, &truth, None);
    }
}
