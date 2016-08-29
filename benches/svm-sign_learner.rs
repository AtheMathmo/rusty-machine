#![feature(test)]

extern crate rusty_machine;
extern crate test;

use rusty_machine::learning::svm::SVM;
// Necessary for the training trait.
use rusty_machine::learning::SupModel;
use rusty_machine::learning::toolkit::kernel::HyperTan;

use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

use test::{Bencher, black_box};

// Sign learner:
//   * Model input a float number
//   * Model output: A float representing the input sign.
//       If the input is positive, the output is close to 1.0.
//       If the input is negative, the output is close to -1.0.
//   * Model generated with the SVM API.
#[bench]
fn bench_svm_sign_learner(b: &mut Bencher) {
    // Training data
    let inputs = Matrix::new(11, 1, vec![
                             -0.1, -2., -9., -101., -666.7,
                             0., 0.1, 1., 11., 99., 456.7
                             ]);
    let targets = Vector::new(vec![
                              -1., -1., -1., -1., -1.,
                              1., 1., 1., 1., 1., 1.
                              ]);

    // Trainee
    b.iter(|| {
        let mut svm_mod = black_box(SVM::new(HyperTan::new(100., 0.), 0.3));
        svm_mod.train(&inputs, &targets);
        for n in (-1000..1000).filter(|&x| x % 100 == 0) {
            let nf = n as f64;
            let input = Matrix::new(1, 1, vec![nf]);
            let _ = black_box(svm_mod.predict(&input));
        }
    });
}
