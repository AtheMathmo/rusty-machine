//! Cross-validation

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
    let folds = Folds::new(inputs.rows(), k);

    let mut costs: Vec<f64> = Vec::new();

    for partition in folds {

        let train_inputs = copy_rows(&inputs, &partition.train_indices);
        let train_targets = copy_rows(&targets, &partition.train_indices);
        let test_inputs = copy_rows(&inputs, &partition.test_indices);
        let test_targets = copy_rows(&targets, &partition.test_indices);

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
fn copy_rows(mat: &Matrix<f64>, rows: &[usize]) -> Matrix<f64> {
    // get the data. copy the bits we're interested in.
    // create a new matrix from this data
    let mut data = vec![0f64; rows.len() * mat.cols()];
    Matrix::<f64>::new(rows.len(), mat.cols(), data)
}

struct Partition {
    train_indices: Vec<usize>,
    test_indices: Vec<usize>
}

struct Folds {
    num_folds: usize,
    indices: Vec<usize>,
    current: Partition,
    count: usize
}

// document the fact that we don't require k to divide sample size,
// and that we round the test set size down
impl Folds {
    fn new(num_samples: usize, num_folds: usize) -> Folds {
        assert!(num_folds > 0 && num_samples >= num_folds,
            "Require num_folds > 0 && num_samples >= num_folds");

        let test_set_size = num_samples / num_folds;
        let train_set_size = num_samples - test_set_size;

        let mut indices: Vec<usize> = (0..num_samples).collect();
        in_place_fisher_yates(&mut indices);

        Folds {
            num_folds: num_folds,
            indices: indices,
            current: Partition {
                train_indices: vec![0; train_set_size],
                test_indices: vec![0; test_set_size]
            },
            count: 0
        }
    }
}

impl Iterator for Folds {
    type Item = Partition;

    fn next(&mut self) -> Option<Partition> {
        if self.count >= self.num_folds {
            return None;
        }

        // TODO: implement!
        self.count += 1;

        None
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
