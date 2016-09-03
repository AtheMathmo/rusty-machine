use rm::learning::optim::Optimizable;
use rm::learning::optim::fmincg::ConjugateGD;
use rm::learning::optim::grad_desc::{GradientDesc, StochasticGD, AdaGrad, RMSProp};
use rm::learning::optim::OptimAlgorithm;

use rm::linalg::Matrix;

/// A model which uses the cost function
/// y = (x - c)^2
///
/// The goal is to learn the true value c which minimizes the cost.
struct XSqModel {
    c: f64,
}

impl Optimizable for XSqModel {
    type Inputs = Matrix<f64>;
	type Targets = Matrix<f64>;

    fn compute_grad(&self, params: &[f64], _: &Matrix<f64>, _: &Matrix<f64>) -> (f64, Vec<f64>) {

        ((params[0] - self.c) * (params[0] - self.c),
         vec![2f64 * (params[0] - self.c)])
    }
}

#[test]
fn convex_fmincg_training() {
    let x_sq = XSqModel { c: 20f64 };

    let cgd = ConjugateGD::default();
    let test_data = vec![500f64];
    let params = cgd.optimize(&x_sq,
                              &test_data[..],
                              &Matrix::zeros(1, 1),
                              &Matrix::zeros(1, 1));

    assert!(params[0] - 20f64 < 1e-10);
    assert!(x_sq.compute_grad(&params, &Matrix::zeros(1, 1), &Matrix::zeros(1, 1)).0 < 1e-10);
}

#[test]
fn convex_gd_training() {
    let x_sq = XSqModel { c: 20f64 };

    let gd = GradientDesc::default();
    let test_data = vec![500f64];
    let params = gd.optimize(&x_sq,
                              &test_data[..],
                              &Matrix::zeros(1, 1),
                              &Matrix::zeros(1, 1));

    assert!(params[0] - 20f64 < 1e-10);
    assert!(x_sq.compute_grad(&params, &Matrix::zeros(1, 1), &Matrix::zeros(1, 1)).0 < 1e-10);
}

#[test]
fn convex_stochastic_gd_training() {
    let x_sq = XSqModel { c: 20f64 };

    let gd = StochasticGD::new(0.5f64, 1f64, 100);
    let test_data = vec![100f64];
    let params = gd.optimize(&x_sq,
                              &test_data[..],
                              &Matrix::zeros(100, 1),
                              &Matrix::zeros(100, 1));

    assert!(params[0] - 20f64 < 1e-10);
    assert!(x_sq.compute_grad(&params, &Matrix::zeros(1, 1), &Matrix::zeros(1, 1)).0 < 1e-10);
}

#[test]
fn convex_adagrad_training() {
    let x_sq = XSqModel { c: 20f64 };

    let gd = AdaGrad::new(5f64, 1f64, 100);
    let test_data = vec![100f64];
    let params = gd.optimize(&x_sq,
                              &test_data[..],
                              &Matrix::zeros(100, 1),
                              &Matrix::zeros(100, 1));

    assert!(params[0] - 20f64 < 1e-10);
    assert!(x_sq.compute_grad(&params, &Matrix::zeros(1, 1), &Matrix::zeros(1, 1)).0 < 1e-10);
}

#[test]
fn convex_rmsprop_training() {
  let x_sq = XSqModel { c: 20f64 };

  let rms = RMSProp::new(0.05, 0.9, 1e-5, 50);
  let test_data = vec![100f64];
  let params = rms.optimize(&x_sq,
                              &test_data[..],
                              &Matrix::zeros(100, 1),
                              &Matrix::zeros(100, 1));

  assert!(params[0] - 20f64 < 1e-10);
  assert!(x_sq.compute_grad(&params, &Matrix::zeros(1, 1), &Matrix::zeros(1, 1)).0 < 1e-10);
}