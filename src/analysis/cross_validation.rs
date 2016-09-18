//! Module for performing cross-validation of models.

use std::cmp;
use std::iter::Chain;
use std::slice::Iter;
use linalg::Matrix;
use learning::{LearningResult, SupModel};
use learning::toolkit::rand_utils::in_place_fisher_yates;

/// Randomly splits the inputs into k 'folds'. For each fold a model
/// is trained using all inputs except for that fold, and tested on the
/// data in the fold. Returns the scores for each fold.
///

/// # Arguments
/// * `model` - Used to train and predict for each fold.
/// * `inputs` - All input samples.
/// * `targets` - All targets.
/// * `k` - Number of folds to use.
/// * `score` - Used to compare the outputs for each fold to the targets. Higher scores are better. See the `analysis::score` module for examples.
///
/// # Examples
/// ```
/// use rusty_machine::analysis::cross_validation::k_fold_validate;
/// use rusty_machine::analysis::score::row_accuracy;
/// use rusty_machine::learning::naive_bayes::{NaiveBayes, Bernoulli};
/// use rusty_machine::linalg::Matrix;
///
/// let inputs = Matrix::new(3, 2, vec![1.0, 1.1,
///                                     5.2, 4.3,
///                                     6.2, 7.3]);
///
/// let targets = Matrix::new(3, 3, vec![1.0, 0.0, 0.0,
///                                      0.0, 0.0, 1.0,
///                                      0.0, 0.0, 1.0]);
///
/// let mut model = NaiveBayes::<Bernoulli>::new();
///
/// let accuracy = k_fold_validate(
///     &mut model,
///     &inputs,
///     &targets,
///     3,
///     row_accuracy
/// );
/// ```
pub fn k_fold_validate<M, S>(model: &mut M,
                             inputs: &Matrix<f64>,
                             targets: &Matrix<f64>,
                             k: usize,
                             score: S) -> LearningResult<Vec<f64>>
    where S: Fn(&Matrix<f64>, &Matrix<f64>) -> f64,
          M: SupModel<Matrix<f64>, Matrix<f64>>,
{
    assert_eq!(inputs.rows(), targets.rows());
    let num_samples = inputs.rows();
    let shuffled_indices = create_shuffled_indices(num_samples);
    let folds = Folds::new(&shuffled_indices, k);

    let mut costs: Vec<f64> = Vec::new();

    for p in folds {
        let train_inputs = copy_rows(&inputs, p.train_indices.clone());
        let train_targets = copy_rows(&targets, p.train_indices.clone());
        let test_inputs = copy_rows(&inputs, p.test_indices.clone());
        let test_targets = copy_rows(&targets, p.test_indices.clone());

        let _ = try!(model.train(&train_inputs, &train_targets));
        let outputs = try!(model.predict(&test_inputs));
        costs.push(score(&outputs, &test_targets));
    }

    Ok(costs)
}

// TODO: Use a preallocated buffer for each fold.
fn copy_rows<'a, I>(mat: &Matrix<f64>,
                    rows: I) -> Matrix<f64>
    where I: ExactSizeIterator<Item=&'a usize>
{
    let num_rows = rows.len();
    let mut data = vec![0f64; num_rows * mat.cols()];
    let mut idx = 0;
    for &row in rows {
        for col in 0..mat.cols(){
            data[idx] = mat[[row, col]];
            idx += 1;
        }
    }
    Matrix::<f64>::new(num_rows, mat.cols(), data)
}

/// A permutation of 0..n.
struct ShuffledIndices(Vec<usize>);

/// Permute the indices of the inputs samples.
fn create_shuffled_indices(num_samples: usize) -> ShuffledIndices {
    let mut indices: Vec<usize> = (0..num_samples).collect();
    in_place_fisher_yates(&mut indices);
    ShuffledIndices(indices)
}

/// A partition of indices of all available samples into
/// a training set and a test set.
struct Partition<'a> {
    train_indices: TrainingIndices<'a>,
    test_indices: TestIndices<'a>
}

#[derive(Clone)]
struct TestIndices<'a>(Iter<'a, usize>);

#[derive(Clone)]
struct TrainingIndices<'a> {
    chain: Chain<Iter<'a, usize>, Iter<'a, usize>>,
    size: usize
}

impl<'a> TestIndices<'a> {
    fn new(indices: &'a [usize]) -> TestIndices<'a> {
        TestIndices(indices.iter())
    }
}

impl<'a> Iterator for TestIndices<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<&'a usize> {
        self.0.next()
    }
}

impl <'a> ExactSizeIterator for TestIndices<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a> TrainingIndices<'a> {
    fn new(left: &'a [usize], right: &'a [usize]) -> TrainingIndices<'a> {
        let chain = left.iter().chain(right.iter());
        TrainingIndices {
            chain: chain,
            size: left.len() + right.len()
        }
    }
}

impl<'a> Iterator for TrainingIndices<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<&'a usize> {
        self.chain.next()
    }
}

impl <'a> ExactSizeIterator for TrainingIndices<'a> {
    fn len(&self) -> usize {
        self.size
    }
}

/// An iterator over the sets of indices required for k-fold cross validation.
struct Folds<'a> {
    num_folds: usize,
    indices: &'a[usize],
    count: usize
}

impl<'a> Folds<'a> {
    /// Let n = indices.len(), and k = num_folds.
    /// The first n % k folds have size n / k + 1 and the
    /// rest have size n / k. (In particular, if n % k == 0 then all
    /// folds are the same size.)
    fn new(indices: &'a ShuffledIndices, num_folds: usize) -> Folds<'a> {
        let num_samples = indices.0.len();
        assert!(num_folds > 1 && num_samples >= num_folds,
            "Require num_folds > 1 && num_samples >= num_folds");

        Folds {
            num_folds: num_folds,
            indices: &indices.0,
            count: 0
        }
    }
}

impl<'a> Iterator for Folds<'a> {
    type Item = Partition<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.num_folds {
            return None;
        }

        let num_samples = self.indices.len();
        let q = num_samples / self.num_folds;
        let r = num_samples % self.num_folds;
        let fold_start = self.count * q + cmp::min(self.count, r);
        let fold_size = if self.count >= r {q} else {q + 1};
        let fold_end = fold_start + fold_size;

        self.count += 1;

        let prefix = &self.indices[..fold_start];
        let suffix = &self.indices[fold_end..];
        let infix = &self.indices[fold_start..fold_end];
        Some(Partition {
            train_indices: TrainingIndices::new(prefix, suffix),
            test_indices: TestIndices::new(infix)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{copy_rows, ShuffledIndices, Folds};
    use linalg::Matrix;

    #[test]
    fn test_copy_rows() {
        let m = Matrix::new(4, 2, vec![ 0.0,  1.0,
                                       10.0, 11.0,
                                       20.0, 21.0,
                                       30.0, 31.0]);

        let s = copy_rows(&m, vec![0, 2].iter());

        assert_eq!(s, Matrix::new(2, 2, vec![ 0.0,  1.0,
                                             20.0, 21.0]));
    }

    // k % n == 0
    #[test]
    fn test_folds_n6_k3() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3, 4, 5]);
        let folds = collect_folds(Folds::new(&idxs, 3));

        assert_eq!(folds, vec![
            (vec![2, 3, 4, 5], vec![0, 1]),
            (vec![0, 1, 4, 5], vec![2, 3]),
            (vec![0, 1, 2, 3], vec![4, 5])
            ]);
    }

    // k % n == 1
    #[test]
    fn test_folds_n5_k2() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3, 4]);
        let folds = collect_folds(Folds::new(&idxs, 2));

        assert_eq!(folds, vec![
            (vec![3, 4], vec![0, 1, 2]),
            (vec![0, 1, 2], vec![3, 4])
            ]);
    }

    // k % n == 2
    #[test]
    fn test_folds_n6_k4() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3, 4, 5]);
        let folds = collect_folds(Folds::new(&idxs, 4));

        assert_eq!(folds, vec![
            (vec![2, 3, 4, 5], vec![0, 1]),
            (vec![0, 1, 4, 5], vec![2, 3]),
            (vec![0, 1, 2, 3, 5], vec![4]),
            (vec![0, 1, 2, 3, 4], vec![5])
            ]);
    }

    // k == n
    #[test]
    fn test_folds_n4_k4() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3]);
        let folds = collect_folds(Folds::new(&idxs, 4));

        assert_eq!(folds, vec![
            (vec![1, 2, 3], vec![0]),
            (vec![0, 2, 3], vec![1]),
            (vec![0, 1, 3], vec![2]),
            (vec![0, 1, 2], vec![3])
            ]);
    }

    #[test]
    #[should_panic]
    fn test_folds_rejects_large_k() {
        let idxs = ShuffledIndices(vec![0, 1, 2]);
        let _ = collect_folds(Folds::new(&idxs, 4));
    }

    // Check we're really returning iterators into the shuffled
    // indices rather than into (0..n).
    #[test]
    fn test_folds_unordered_indices() {
        let idxs = ShuffledIndices(vec![5, 4, 3, 2, 1, 0]);
        let folds = collect_folds(Folds::new(&idxs, 3));

        assert_eq!(folds, vec![
            (vec![3, 2, 1, 0], vec![5, 4]),
            (vec![5, 4, 1, 0], vec![3, 2]),
            (vec![5, 4, 3, 2], vec![1, 0])
            ]);
    }

    fn collect_folds<'a>(folds: Folds<'a>) -> Vec<(Vec<usize>, Vec<usize>)> {
        folds
            .map(|p|
                (p.train_indices.map(|x| *x).collect::<Vec<_>>(),
                 p.test_indices.map(|x| *x).collect::<Vec<_>>()))
            .collect::<Vec<(Vec<usize>, Vec<usize>)>>()
    }
}
