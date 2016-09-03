//! Cross-validation

use std::iter::Chain;
use std::slice::Iter;
use linalg::Matrix;
use learning::SupModel;
use learning::toolkit::cost_fn::*;
use learning::toolkit::rand_utils::in_place_fisher_yates;

// TODO: Support other input and output types. To do this
// TODO: we'd need to add an Iterator<SomeRowType> bound, and
// TODO: check that Matrix implements Iterator.
// TODO:
// TODO: More importantly: DON'T COPY THE DATA FOR EACH FOLD
// TODO: See comment on copy_rows.
// TODO:
// TODO: Remove gradient from CostFunc and add a new trait
// TODO: for differentiable cost functions.
/// Randomly splits the inputs into k 'folds'. For each fold a model
/// is trained using all inputs except for that fold, and tested on the
/// data in the fold. Returns the mean cost.
pub fn k_fold_validate<M, C>(model: &mut M,
                             inputs: &Matrix<f64>,
                             targets: &Matrix<f64>,
                             k: usize) -> f64
    where C: CostFunc<Matrix<f64>>,
          M: SupModel<Matrix<f64>, Matrix<f64>>,
{
    assert_eq!(inputs.rows(), targets.rows());
    let num_samples = inputs.rows();
    let shuffled_indices = create_shuffled_indices(num_samples);
    let folds = Folds::new(&shuffled_indices, k);

    let mut costs: Vec<f64> = Vec::new();

    for p in folds {
        let test_size = p.test_indices.len();
        let train_size = num_samples - test_size;
        let train_inputs = copy_rows(&inputs, p.train_indices.clone(), train_size);
        let train_targets = copy_rows(&targets, p.train_indices.clone(), train_size);
        let test_inputs = copy_rows(&inputs, p.test_indices.clone(), test_size);
        let test_targets = copy_rows(&targets, p.test_indices.clone(), test_size);

        let cost = train_and_test::<Matrix<f64>, Matrix<f64>, M, C>(
            model,
            &train_inputs,
            &train_targets,
            &test_inputs,
            &test_targets);

        costs.push(cost);
    }

    costs.into_iter().fold(0f64, |acc, c| acc + c) / (k as f64)
}

// TODO: Don't copy! Is there support for matrix views whose
// TODO: rows are not contiguous in the original matrix?
// TODO: If not, should we change some signatures defined in terms
// TODO: Matrix<f64> to be instead use iterators of &[f64]s?
/// We need to pass num_rows separately as Partition.train_indices
/// is a Chain, and this doesn't implement ExactSizeIterator:
/// https://github.com/rust-lang/rust/issues/34433
fn copy_rows<'a, I>(mat: &Matrix<f64>,
                    rows: I,
                    num_rows: usize) -> Matrix<f64>
    where I: Iterator<Item=&'a usize>
{
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

/// A partition of 0..n into a training set and a test set.
struct Partition<'a> {
    train_indices: Chain<Iter<'a, usize>, Iter<'a, usize>>,
    test_indices: Iter<'a, usize>
}

/// An iterator over the Partitions required for k-fold cross validation.
struct Folds<'a> {
    num_folds: usize,
    fold_size: usize,
    indices: &'a[usize],
    count: usize
}

/// A permutation of 0..n
struct ShuffledIndices(Vec<usize>);

/// Iterating over folds produces views into a single block of data.
/// As that data can't be owned by the Folds instance itself, we have
/// to create the shuffled indices here and then pass this to Folds::new.
fn create_shuffled_indices(num_samples: usize) -> ShuffledIndices {
    let mut indices: Vec<usize> = (0..num_samples).collect();
    in_place_fisher_yates(&mut indices);
    ShuffledIndices(indices)
}

impl<'a> Folds<'a> {
    fn new(indices: &'a ShuffledIndices, num_folds: usize) -> Folds<'a> {
        let num_samples = indices.0.len();
        assert!(num_folds > 1 && num_samples >= num_folds,
            "Require num_folds > 1 && num_samples >= num_folds");
        assert!(num_samples % num_folds == 0,
            "Require num_samples % num_folds == 0");

        Folds {
            num_folds: num_folds,
            fold_size: num_samples / num_folds,
            indices: &indices.0,
            count: 0
        }
    }

    /// Create a partition which uses the kth fold as test set.
    /// Panics if (0-indexed) k is out of bounds.
    fn create_partition(&self, k: usize) -> Partition<'a> {
        assert!(k < self.num_folds);

        // Test on data within the fold, train on the rest
        let fold_start = k * self.fold_size;
        let fold_end = fold_start + self.fold_size;

        let prefix = &self.indices[..fold_start];
        let suffix = &self.indices[fold_end..];
        let infix = self.indices[fold_start..fold_end].iter();

        Partition {
            train_indices: prefix.iter().chain(suffix.iter()),
            test_indices: infix
        }
    }
}

impl<'a> Iterator for Folds<'a> {
    type Item = Partition<'a>;

    fn next(&mut self) -> Option<Partition<'a>> {
        if self.count >= self.num_folds {
            return None;
        }
        let partition = self.create_partition(self.count);
        self.count += 1;
        Some(partition)
    }
}

/// Docs go here
pub fn train_and_test<I, T, M, C>(model: &mut SupModel<I, T>,
                                  train_inputs: &I,
                                  train_targets: &T,
                                  test_inputs: &I,
                                  test_targets: &T)
                                  -> f64
    where C: CostFunc<T>
{
    model.train(train_inputs, train_targets);
    let outputs = model.predict(test_inputs);
    C::cost(&outputs, test_targets)
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

        let s = copy_rows(&m, vec![0, 2].iter(), 2);

        assert_eq!(s, Matrix::new(2, 2, vec![ 0.0,  1.0,
                                             20.0, 21.0]));
    }

    #[test]
    fn test_folds() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3, 4, 5]);
        let folds: Vec<(Vec<usize>, Vec<usize>)> =
            Folds::new(&idxs, 3)
                .map(|p|
                    (p.train_indices.map(|x| *x).collect::<Vec<_>>(),
                     p.test_indices.map(|x| *x).collect::<Vec<_>>()))
                .collect();

        assert_eq!(folds, vec![
            (vec![2, 3, 4, 5], vec![0, 1]),
            (vec![0, 1, 4, 5], vec![2, 3]),
            (vec![0, 1, 2, 3], vec![4, 5])
            ]);
    }
}
