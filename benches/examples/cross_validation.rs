use rusty_machine::linalg::{Matrix, BaseMatrix};
use rusty_machine::learning::{LearningResult, SupModel};
use rusty_machine::analysis::score::row_accuracy;
use rusty_machine::analysis::cross_validation::k_fold_validate;
use rand::{thread_rng, Rng};
use test::{Bencher, black_box};

fn generate_data(rows: usize, cols: usize) -> Matrix<f64> {
    let mut rng = thread_rng();
    let mut data = Vec::with_capacity(rows * cols);

    for _ in 0..data.capacity() {
        data.push(rng.gen_range(0f64, 1f64));
    }

    Matrix::new(rows, cols, data)
}

/// A very simple model that looks at all the data it's
/// given but doesn't do anything useful.
/// Stores the sum of all elements in the inputs and targets
/// matrices when trained. Its prediction for each row is the
/// sum of the row's elements plus the precalculated training sum.
struct DummyModel {
    sum: f64
}

impl SupModel<Matrix<f64>, Matrix<f64>> for DummyModel {
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        let predictions: Vec<f64> = inputs
            .row_iter()
            .map(|row| { self.sum + sum(row.iter()) })
            .collect();
        Ok(Matrix::new(inputs.rows(), 1, predictions))
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Matrix<f64>) -> LearningResult<()> {
        self.sum = sum(inputs.iter()) + sum(targets.iter());
        Ok(())
    }
}

fn sum<'a, I: Iterator<Item=&'a f64>>(x: I) -> f64 {
    x.fold(0f64, |acc, x| acc + x)
}

macro_rules! bench {
    ($name:ident: $params:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let (rows, cols, k) = $params;
            let inputs = generate_data(rows, cols);
            let targets = generate_data(rows, 1);

            b.iter(|| {
                let mut model = DummyModel { sum: 0f64 };
                let _ = black_box(
                    k_fold_validate(&mut model, &inputs, &targets, k, row_accuracy)
                );
            });
        }
    }
}

bench!(bench_10_10_3: (10, 10, 3));
bench!(bench_1000_10_3: (1000, 10, 3));
bench!(bench_1000_10_10: (1000, 10, 10));
bench!(bench_1000_10_100: (1000, 10, 100));
