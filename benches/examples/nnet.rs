use test::{Bencher, black_box};

use rand::{random, Closed01};
use std::vec::Vec;

use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
use rusty_machine::learning::optim::grad_desc::StochasticGD;

use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;

fn generate_data() -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    const THRESHOLD: f64 = 0.7;
    const SAMPLES: usize = 1000;

    let mut input_data = Vec::with_capacity(SAMPLES * 2);
    let mut label_data = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        // The two inputs are "signals" between 0 and 1
        let Closed01(left) = random::<Closed01<f64>>();
        let Closed01(right) = random::<Closed01<f64>>();
        input_data.push(left);
        input_data.push(right);
        if left > THRESHOLD && right > THRESHOLD {
            label_data.push(1.0);
        } else {
            label_data.push(0.0)
        }
    }

    let inputs = Matrix::new(SAMPLES, 2, input_data);
    let targets = Matrix::new(SAMPLES, 1, label_data);

    let test_cases = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
        ];
    let test_inputs = Matrix::new(test_cases.len() / 2, 2, test_cases);

    (inputs, targets, test_inputs)
}

#[bench]
fn nnet_and_gate_train(b: &mut Bencher) {
    let (inputs, targets, _) = generate_data();
    let layers = &[2, 1];
    let criterion = BCECriterion::new(Regularization::L2(0.));

    b.iter(|| {
        let mut model = black_box(NeuralNet::mlp(layers, criterion, StochasticGD::default(), Sigmoid));
        let _ = black_box(model.train(&inputs, &targets).unwrap());
    })
}

#[bench]
fn nnet_and_gate_predict(b: &mut Bencher) {
    let (inputs, targets, test_inputs) = generate_data();
    let layers = &[2, 1];
    let criterion = BCECriterion::new(Regularization::L2(0.));

    let mut model = NeuralNet::mlp(layers, criterion, StochasticGD::default(), Sigmoid);
    let _ = model.train(&inputs, &targets);

    b.iter(|| {
        let _ = black_box(model.predict(&test_inputs));
    })
}
