extern crate rusty_machine;
extern crate rand;

use rand::Rand;
use rand::distributions::Sample;
use rand::distributions::normal::Normal;
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};
use rusty_machine::linalg::{Matrix, BaseMatrix};
use rusty_machine::learning::SupModel;


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
    /// Generate a random dog.
    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        // Friendliness, furriness, and speed are normally distributed and
        // (given color:) independent.
        let mut red_dog_friendliness = Normal::new(0., 1.);
        let mut red_dog_furriness = Normal::new(0., 1.);
        let mut red_dog_speed = Normal::new(0., 1.);

        let mut white_dog_friendliness = Normal::new(1., 1.);
        let mut white_dog_furriness = Normal::new(1., 1.);
        let mut white_dog_speed = Normal::new(-1., 1.);

        // Flip a coin to decide whether to generate a red or white dog.
        let coin: f64 = rng.gen();
        let color = if coin < 0.5 { Color::Red } else { Color::White };

        match color {
            Color::Red => {
                Dog {
                    color: Color::Red,
                    // sample from our normal distributions for each trait
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

fn generate_dog_data(training_set_size: u32, test_set_size: u32)
    -> (Matrix<f64>, Matrix<f64>, Matrix<f64>, Vec<Dog>) {
    let mut randomness = rand::StdRng::new()
        .expect("we should be able to get an RNG");
    let rng = &mut randomness;

    // We'll train the model on these dogs
    let training_dogs = (0..training_set_size)
        .map(|_| { Dog::rand(rng) })
        .collect::<Vec<_>>();

    // ... and then use the model to make predictions about these dogs' color
    // given only their trait measurements.
    let test_dogs = (0..test_set_size)
        .map(|_| { Dog::rand(rng) })
        .collect::<Vec<_>>();

    // The model's `.train` method will take two matrices, each with a row for
    // each dog in the training set: the rows in the first matrix contain the
    // trait measurements; the rows in the second are either [1, 0] or [0, 1]
    // to indicate color.
    let training_data: Vec<f64> = training_dogs.iter()
        .flat_map(|dog| vec![dog.friendliness, dog.furriness, dog.speed])
        .collect();
    let training_matrix: Matrix<f64> = training_data.chunks(3).collect();
    let target_data: Vec<f64> = training_dogs.iter()
        .flat_map(|dog| match dog.color {
            Color::Red => vec![1., 0.],
            Color::White => vec![0., 1.],
        })
        .collect();
    let target_matrix: Matrix<f64> = target_data.chunks(2).collect();

    // Build another matrix for the test set of dogs to make predictions about.
    let test_data: Vec<f64> = test_dogs.iter()
        .flat_map(|dog| vec![dog.friendliness, dog.furriness, dog.speed])
        .collect();
    let test_matrix: Matrix<f64> = test_data.chunks(3).collect();

    (training_matrix, target_matrix, test_matrix, test_dogs)
}

fn evaluate_prediction(hits: &mut u32, dog: &Dog, prediction: &[f64]) -> (Color, bool) {
    let predicted_color = dog.color;
    let actual_color = if prediction[0] == 1. {
        Color::Red
    } else {
        Color::White
    };
    let accurate = predicted_color == actual_color;
    if accurate {
        *hits += 1;
    }
    (actual_color, accurate)
}

fn main() {
    let (training_set_size, test_set_size) = (1000, 1000);
    // Generate all of our train and test data
    let (training_matrix, target_matrix, test_matrix, test_dogs) = generate_dog_data(training_set_size, test_set_size);

    // Train!
    let mut model = NaiveBayes::<naive_bayes::Gaussian>::new();
    model.train(&training_matrix, &target_matrix)
        .expect("failed to train model of dogs");

    // Predict!
    let predictions = model.predict(&test_matrix)
        .expect("failed to predict dogs!?");

    // Score how well we did.
    let mut hits = 0;
    let unprinted_total = test_set_size.saturating_sub(10) as usize;
    for (dog, prediction) in test_dogs.iter().zip(predictions.row_iter()).take(unprinted_total) {
        evaluate_prediction(&mut hits, dog, prediction.raw_slice());
    }
    
    if unprinted_total > 0 {
        println!("...");
    }
    
    for (dog, prediction) in test_dogs.iter().zip(predictions.row_iter()).skip(unprinted_total) {
        let (actual_color, accurate) = evaluate_prediction(&mut hits, dog, prediction.raw_slice());
        println!("Predicted: {:?}; Actual: {:?}; Accurate? {:?}",
                 dog.color, actual_color, accurate);
    }

    println!("Accuracy: {}/{} = {:.1}%", hits, test_set_size,
             (f64::from(hits))/(f64::from(test_set_size)) * 100.);
}
