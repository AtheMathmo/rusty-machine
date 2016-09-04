extern crate rusty_machine;

use rusty_machine::learning::naive_bayes::{NaiveBayes, Bernoulli};
use rusty_machine::learning::toolkit::cross_validation::k_fold_validate;
use rusty_machine::learning::toolkit::cost_fn::{MeanSqError};
use rusty_machine::linalg::Matrix;

fn main () {
    let inputs = Matrix::new(6, 2, vec![1.0, 1.1,
                                        1.1, 0.9,
                                        2.2, 2.3,
                                        2.5, 2.7,
                                        5.2, 4.3,
                                        6.2, 7.3]);

    let targets = Matrix::new(6,3, vec![1.0, 0.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0,
                                        0.0, 0.0, 1.0]);

    let mut model = NaiveBayes::<Bernoulli>::new();
    let k = 3;

    let costs = k_fold_validate::<NaiveBayes<Bernoulli>, MeanSqError>(
        &mut model,
        &inputs,
        &targets,
        k
    );

    println!("Costs are: {:?}", costs);
}
