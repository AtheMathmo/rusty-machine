//! Cross-validation

use std::iter::Chain;
use std::slice::Iter;
use linalg::Matrix;
use learning::SupModel;
use learning::toolkit::cost_fn::*;
use learning::toolkit::rand_utils::in_place_fisher_yates;

// TODO: Support other input and output types.
// TODO:
// TODO: DON'T ALLOCATE DATA FOR EACH FOLD
// TODO: See comment on copy_rows.
// TODO:
// TODO: Remove gradient from CostFunc and add a new trait
// TODO: for differentiable cost functions.
// TODO:
// TODO: Clarify what happens when model.train is called multiple
// TODO: times. This assumes that it throws away the old data and
// TODO: trains a new model.
/// Randomly splits the inputs into k 'folds'. For each fold a model
/// is trained using all inputs except for that fold, and tested on the
/// data in the fold. Returns the costs for each fold.
pub fn k_fold_validate<M, C>(model: &mut M,
                             inputs: &Matrix<f64>,
                             targets: &Matrix<f64>,
                             k: usize) -> Vec<f64>
    where C: CostFunc<Matrix<f64>>,
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

        model.train(&train_inputs, &train_targets);
        let outputs = model.predict(&test_inputs);
        costs.push(C::cost(&outputs, &test_targets));
    }

    costs
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
    fold_size: usize,
    indices: &'a[usize],
    count: usize
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
}

impl<'a> Iterator for Folds<'a> {
    type Item = Partition<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.num_folds {
            return None;
        }

        let fold_start = self.count * self.fold_size;
        let fold_end = fold_start + self.fold_size;
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

    #[test]
    fn test_folds_ordered_indices() {
        let idxs = ShuffledIndices(vec![0, 1, 2, 3, 4, 5]);
        let folds = collect_folds(Folds::new(&idxs, 3));

        assert_eq!(folds, vec![
            (vec![2, 3, 4, 5], vec![0, 1]),
            (vec![0, 1, 4, 5], vec![2, 3]),
            (vec![0, 1, 2, 3], vec![4, 5])
            ]);
    }

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
