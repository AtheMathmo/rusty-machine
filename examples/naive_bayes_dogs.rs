extern crate rusty_machine;
extern crate rand;

use rand::Rand;
use rand::distributions::Sample;
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};
use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;
use rusty_machine::stats::dist::gaussian::Gaussian;


#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Color {
    Red,
    White,
}

#[derive(Clone, Debug)]
struct Dog {
    color: Color,
    friendliness: f64,
    furriness: f64,
    speed: f64,
}

impl Rand for Dog {
    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        let mut red_dog_friendliness = Gaussian::from_std_dev(0., 1.);
        let mut red_dog_furriness = Gaussian::from_std_dev(0., 1.);
        let mut red_dog_speed = Gaussian::from_std_dev(0., 1.);

        let mut white_dog_friendliness = Gaussian::from_std_dev(1., 1.);
        let mut white_dog_furriness = Gaussian::from_std_dev(1., 1.);
        let mut white_dog_speed = Gaussian::from_std_dev(-1., 1.);

        let coin: f64 = rng.gen();
        let color = if coin < 0.5 { Color::Red } else { Color::White };

        match color {
            Color::Red => {
                Dog {
                    color: Color::Red,
                    friendliness: red_dog_friendliness.sample(rng),
                    furriness: red_dog_furriness.sample(rng),
                    speed: red_dog_speed.sample(rng),
                }
            },
            Color::White => {
                Dog {
                    color: Color::White,
                    friendliness: white_dog_friendliness.sample(rng),
                    furriness: white_dog_furriness.sample(rng),
                    speed: white_dog_speed.sample(rng),
                }
            },
        }
    }
}

fn main() {
    let mut randomness = rand::StdRng::new()
        .expect("we should be able to get an RNG");
    let rng = &mut randomness;

    let training_set_size = 1000;
    let test_set_size = 1000;

    let training_dogs = (0..training_set_size)
        .map(|_| { Dog::rand(rng) })
        .collect::<Vec<_>>();

    let test_dogs = (0..test_set_size)
        .map(|_| { Dog::rand(rng) })
        .collect::<Vec<_>>();

    let mut training_matrix = Matrix::new(0, 3, Vec::new());
    let mut target_matrix = Matrix::new(0, 2, Vec::new());
    for training_dog in &training_dogs {
        let dog_row = Matrix::new(
            1, 3,
            vec![training_dog.friendliness,
                 training_dog.furriness,
                 training_dog.speed]
        );
        training_matrix = training_matrix.vcat(&dog_row);
        let color_row = match training_dog.color {
            Color::Red => Matrix::new(1, 2, vec![1., 0.]),
            Color::White => Matrix::new(1, 2, vec![0., 1.]),
        };
        target_matrix = target_matrix.vcat(&color_row);
    }

    let mut model = NaiveBayes::<naive_bayes::Gaussian>::new();
    model.train(&training_matrix, &target_matrix)
        .expect("failed to train model of dogs");

    let mut test_matrix = Matrix::new(0, 3, Vec::new());
    for test_dog in &test_dogs {
        let dog_row = Matrix::new(
            1, 3,
            vec![test_dog.friendliness,
                 test_dog.furriness,
                 test_dog.speed]
        );
        test_matrix = test_matrix.vcat(&dog_row);
    }

    let predictions = model.predict(&test_matrix)
        .expect("failed to predict dogs!?");

    let mut hits = 0;
    for (dog, prediction) in test_dogs.iter().zip(predictions.iter_rows()) {
        let predicted_color = dog.color;
        let actual_color = if prediction[0] == 1. {
            Color::Red
        } else {
            Color::White
        };
        let accurate = predicted_color == actual_color;
        if accurate {
            hits += 1;
        }
        println!("Predicted: {:?}; Actual: {:?}; Accurate? {:?}",
                 predicted_color, actual_color, accurate);
    }

    println!("Accuracy: {}/{} = {:.1}%", hits, test_set_size,
             (hits as f64)/(test_set_size as f64) * 100.);
}
