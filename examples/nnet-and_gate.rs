extern crate rusty_machine;
extern crate rand;

use rand::{random, Closed01};
use std::vec::Vec;

use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::toolkit::activ_fn::Sigmoid;
use rusty_machine::learning::optim::grad_desc::StochasticGD;

use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;

// AND gate
fn main() {
    println!("AND gate learner sample:");

    const THRESHOLD: f64 = 0.7;

    const SAMPLES: usize = 10000;
    println!("Generating {} training data and labels...", SAMPLES as u32);

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
    // Create a multilayer perceptron with an input layer of size 2 and output layer of size 1
    // Uses a Sigmoid activation function and uses Stochastic gradient descent for training
    let mut model = NeuralNet::mlp(layers, criterion, StochasticGD::default(), Sigmoid);

    println!("Training...");
    // Our train function returns a Result<(), E>
    model.train(&inputs, &targets).unwrap();

    let test_cases = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
        ];
    let expected = vec![
        0.0,
        0.0,
        1.0,
        0.0,
        ];
    let test_inputs = Matrix::new(test_cases.len() / 2, 2, test_cases);
    let res = model.predict(&test_inputs).unwrap();

    println!("Evaluation...");
    let mut hits = 0;
    let mut misses = 0;
    // Evaluation
    println!("Got\tExpected");
    for (idx, prediction) in res.into_vec().iter().enumerate() {
        println!("{:.2}\t{}", prediction, expected[idx]);
        if (prediction - 0.5) * (expected[idx] - 0.5) > 0. {
            hits += 1;
        } else {
            misses += 1;
        }
    }

    println!("Hits: {}, Misses: {}", hits, misses);
    let hits_f = hits as f64;
    let total = (hits + misses) as f64;
    println!("Accuracy: {}%", (hits_f / total) * 100.);
}
