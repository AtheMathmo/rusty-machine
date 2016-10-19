extern crate rusty_machine;
extern crate rand;

use rand::distributions::{Normal, IndependentSample};
use rand::{Rng, thread_rng};

use rusty_machine::learning::gmm::{GaussianMixtureModel, Random, CholeskyFull};
use rusty_machine::learning::UnSupModel;
use rusty_machine::linalg::Matrix;

// Simulate some data from gaussian clusters
fn simulate_gmm_1d_data(count: usize,
                        means: Vec<f64>,
                        vars: Vec<f64>,
                        mixture_weights: Vec<f64>)
                        -> Vec<f64> {
    assert_eq!(means.len(), vars.len());
    assert_eq!(means.len(), mixture_weights.len());

    let gmm_count = means.len();

    let mut gaussians = Vec::with_capacity(gmm_count);

    for i in 0..gmm_count {
        // Create a Gaussian with given mean and var
        gaussians.push(Normal::new(means[i], vars[i].sqrt()));
    }

    let mut rng = thread_rng();
    let mut out_samples = Vec::with_capacity(count);

    for _ in 0..count {
        // Pick a gaussian from the mixture weights
        let chosen_gaussian = gaussians[pick_gaussian(&mixture_weights, &mut rng)];

        // Draw sample from it
        let sample = chosen_gaussian.ind_sample(&mut rng);

        // Add to data
        out_samples.push(sample);
    }

    out_samples
}

// A utility function which chooses an index from some mixture weights
fn pick_gaussian<R: Rng>(unnorm_dist: &[f64], rng: &mut R) -> usize {
    assert!(unnorm_dist.len() > 0);
    let sum = unnorm_dist.iter().fold(0f64, |acc, &x| acc + x);
    let rand = rng.gen_range(0.0f64, sum);

    let mut tempsum = 0.0;
    for (i, p) in unnorm_dist.iter().enumerate() {
        tempsum += *p;
        if rand < tempsum {
            return i;
        }
    }

    panic!("No random value was sampled!");
}


fn main() {
    let gmm_count = 3;
    let count = 1000;

    let means = vec![-3f64, 0., 3.];
    let vars = vec![1f64, 0.5, 0.25];
    let weights = vec![0.5, 0.25, 0.25];

    let data = simulate_gmm_1d_data(count, means, vars, weights);

    let mut gmm: GaussianMixtureModel<Random, CholeskyFull> = GaussianMixtureModel::new(gmm_count);
    gmm.set_max_iters(300);
    gmm.train(&Matrix::new(count, 1, data)).unwrap();

    println!("Means = {:?}", gmm.means());
    println!("Covs = {:?}", gmm.covariances());
    println!("Mix Weights = {:?}", gmm.mixture_weights());
}