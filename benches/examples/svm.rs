use rusty_machine::learning::svm::SVM;
// Necessary for the training trait.
use rusty_machine::learning::SupModel;
use rusty_machine::learning::toolkit::kernel::HyperTan;

use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

use test::{Bencher, black_box};

fn generate_data() -> (Matrix<f64>, Vector<f64>) {
    // Training data
    let inputs = Matrix::new(11, 1, vec![
                             -0.1, -2., -9., -101., -666.7,
                             0., 0.1, 1., 11., 99., 456.7
                             ]);
    let targets = Vector::new(vec![
                              -1., -1., -1., -1., -1.,
                              1., 1., 1., 1., 1., 1.
                              ]);

    (inputs, targets)
}

// Sign learner:
//   * Model input a float number
//   * Model output: A float representing the input sign.
//       If the input is positive, the output is close to 1.0.
//       If the input is negative, the output is close to -1.0.
//   * Model generated with the SVM API.
#[bench]
fn svm_sign_learner_train(b: &mut Bencher) {
    let (inputs, targets) = generate_data();

    // Trainee
    b.iter(|| {
        let mut svm_mod = black_box(SVM::new(HyperTan::new(100., 0.), 0.3));
        let _ = black_box(svm_mod.train(&inputs, &targets).unwrap());
    });
}

#[bench]
fn svm_sign_learner_predict(b: &mut Bencher) {
    let (inputs, targets) = generate_data();

    let test_data = (-1000..1000).filter(|&x| x % 100 == 0).map(|x| x as f64).collect::<Vec<_>>();
    let test_inputs = Matrix::new(test_data.len(), 1, test_data);
    let mut svm_mod = SVM::new(HyperTan::new(100., 0.), 0.3);
    let _ = svm_mod.train(&inputs, &targets);
    b.iter(|| {
        let _ = black_box(svm_mod.predict(&test_inputs).unwrap());
    });
}
