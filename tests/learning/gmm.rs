extern crate rand;

use rm::linalg::{Matrix, BaseMatrix};
use rm::learning::gmm::GaussianMixtureModel;
use rm::learning::UnSupModel;

use self::rand::thread_rng;
use self::rand::distributions::IndependentSample;
use self::rand::distributions::normal::Normal;

fn generate_data(centroids: &Matrix<f64>, points_per_centroid: usize, noise: f64) -> Matrix<f64> {
    assert!(centroids.cols() > 0, "Centroids cannot be empty.");
    assert!(centroids.rows() > 0, "Centroids cannot be empty.");
    assert!(noise >= 0f64, "Noise must be non-negative.");
    let mut raw_cluster_data = Vec::with_capacity(centroids.rows() * points_per_centroid *
                                                  centroids.cols());

    let mut rng = thread_rng();
    let normal_rv = Normal::new(0f64, noise);

    for _ in 0..points_per_centroid {
        // Generate points from each centroid
        for centroid in centroids.iter_rows() {
            // Generate a point randomly around the centroid
            let mut point = Vec::with_capacity(centroids.cols());
            for feature in centroid {
                point.push(feature + normal_rv.ind_sample(&mut rng));
            }

            // Push point to raw_cluster_data
            raw_cluster_data.extend(point);
        }
    }

    Matrix::new(centroids.rows() * points_per_centroid,
                centroids.cols(),
                raw_cluster_data)
}

#[test]
fn gmm_train() {

    const SAMPLES_PER_CENTROID: usize = 2000;
    // Choose three cluster centers, at (-0.5, -0.5), (0, 0.5), (0.5, 0.5).
    let centroids = Matrix::new(3, 2, vec![-0.5, -0.5, 0.0, 0.5, 0.5, 0.0]);

    // Generate some data randomly around the centroids
    let samples = generate_data(&centroids, SAMPLES_PER_CENTROID, 0.2);
    let mut model = GaussianMixtureModel::new(3);
    model.set_max_iters(100);
    match model.train(&samples) {
        Ok(_) => {
            println!("means: \n{:.4}", model.means().unwrap());
            println!("log_lik: \n{:.4}", model.log_lik());
            for cov in model.covariances().as_ref().unwrap().iter() {
                println!("cov: \n{:.4}", cov);
            }
        },
        Err(e) => println!("error: {}", e)
    }
}
