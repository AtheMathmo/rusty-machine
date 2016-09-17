extern crate rusty_machine;

use rusty_machine::learning::svm::SVM;
// Necessary for the training trait.
use rusty_machine::learning::SupModel;
use rusty_machine::learning::toolkit::kernel::HyperTan;

use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

// Sign learner:
//   * Model input a float number
//   * Model output: A float representing the input sign.
//       If the input is positive, the output is close to 1.0.
//       If the input is negative, the output is close to -1.0.
//   * Model generated with the SVM API.
fn main() {
    println!("Sign learner sample:");

    println!("Training...");
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
    let mut svm_mod = SVM::new(HyperTan::new(100., 0.), 0.3);
    // Our train function returns a Result<(), E>
    svm_mod.train(&inputs, &targets).unwrap();

    println!("Evaluation...");
    let mut hits = 0;
    let mut misses = 0;
    // Evaluation
    //   Note: We could pass all input values at once to the `predict` method!
    //         Here, we use a loop just to count and print logs.
    for n in (-1000..1000).filter(|&x| x % 100 == 0) {
        let nf = n as f64;
        let input = Matrix::new(1, 1, vec![nf]);
        let out = svm_mod.predict(&input).unwrap();
        let res = if out[0] * nf > 0. {
            hits += 1;
            true
        } else if nf == 0. {
            hits += 1;
            true
        } else {
            misses += 1;
            false
        };

        println!("{} -> {}: {}", Matrix::data(&input)[0], out[0], res);
    }

    println!("Performance report:");
    println!("Hits: {}, Misses: {}", hits, misses);
    let hits_f = hits as f64;
    let total = (hits + misses) as f64;
    println!("Accuracy: {}", (hits_f / total) * 100.);
}
