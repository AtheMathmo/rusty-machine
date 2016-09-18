extern crate rusty_machine;

use rusty_machine::analysis::score::{row_accuracy, neg_mean_squared_error};
use rusty_machine::learning::naive_bayes::{NaiveBayes, Bernoulli};
use rusty_machine::learning::toolkit::cross_validation::k_fold_validate;
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

    let accuracy = k_fold_validate(
        &mut model,
        &inputs,
        &targets,
        k,
        row_accuracy
    );

    println!("Accuracy: {:?}", accuracy);

    let neg_mean_squared_error = k_fold_validate(
        &mut model,
        &inputs,
        &targets,
        k,
        neg_mean_squared_error
    );

    println!("Neg-error: {:?}", neg_mean_squared_error);

}
