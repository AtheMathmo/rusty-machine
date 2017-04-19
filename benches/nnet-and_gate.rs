#![feature(test)]

extern crate rusty_machine;
extern crate rand;
extern crate test;

use test::{Bencher, black_box};

use rand::{random, Closed01};
use std::vec::Vec;

use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::optim::grad_desc::StochasticGD;

use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;

#[bench]
fn bench_nnet_and_gate(b: &mut Bencher) {
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

    let layers = &[2, 1];
    let criterion = BCECriterion::new(Regularization::L2(0.));

    let test_cases = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
        ];
    let test_inputs = Matrix::new(test_cases.len() / 2, 2, test_cases);

    b.iter(|| {
        let mut model = black_box(NeuralNet::new(layers, criterion, StochasticGD::default()));
        model.train(&inputs, &targets);
        let _ = model.predict(&test_inputs);
    })
}
