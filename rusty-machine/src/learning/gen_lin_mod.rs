use linalg::vector::Vector;
use linalg::matrix::Matrix;

use learning::SupModel;

pub struct GenLinearModel<C: Criterion> {
    parameters: Option<Vector<f64>>,
    criterion: C,
}

impl<C: Criterion> GenLinearModel<C> {
	pub fn new(criterion: C) -> GenLinearModel<C> {
		GenLinearModel {
			parameters: None,
			criterion: criterion,
		}
	}
}

impl<C: Criterion> SupModel<Matrix<f64>, Vector<f64>> for GenLinearModel<C> {
    /// Predict output from inputs.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
        if let Some(ref v) = self.parameters {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            self.criterion.apply_link_func(full_inputs * v)
        } else {
            panic!("The model has not been trained.");
        }
    }

    /// Train the model using inputs and targets.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let n = inputs.rows();

        assert!(n == targets.size(),
                "Training data do not have the same dimensions.");

        // Should construct this more sensibly.
        let mut mu = targets.clone();
        let mut z = mu.clone();
        let mut beta : Vector<f64> = Vector::new(vec![]);

        let x_t = inputs.transpose();

        // Iterate to convergence
        for _ in 0..1000 {
            let mut w_diag = Vec::with_capacity(n);
            let mut y_bar_data = Vec::with_capacity(n);

            for (idx, m) in mu.data().iter().enumerate() {
                let link_grad = self.criterion.link_grad(*m);

                y_bar_data.push(link_grad * (targets[idx] - *m));
                w_diag.push(1f64 / (self.criterion.model_variance(*m) * link_grad * link_grad));
            }

            let w = Matrix::from_diag(&w_diag);
            let y_bar = Vector::new(y_bar_data);

            beta = (&x_t * &w * inputs).inverse() * &x_t * w * z;

            // Update z and mu
            let fitted = inputs * &beta;
            z = y_bar + &fitted;
            mu = self.criterion.apply_link_inv(fitted);
        }

        self.parameters = Some(beta);
    }
}

/// Should include the link function and the variance from the distribution
pub trait Criterion {
	type Link : LinkFunc;

    fn model_variance(&self, mu: f64) -> f64;

    fn apply_link_func(&self, vec: Vector<f64>) -> Vector<f64> {
        vec.apply(&Self::Link::func)
    }

    fn apply_link_inv(&self, vec: Vector<f64>) -> Vector<f64> {
    	vec.apply(&Self::Link::func_inv)
    }

    fn link_grad(&self, mu: f64) -> f64 {
        Self::Link::func(mu)
    }
}

pub trait LinkFunc {
    /// The link function.
    fn func(x: f64) -> f64;

    /// The gradient of the link function.
    fn func_grad(x: f64) -> f64;

    /// The inverse of the link function.
    /// Often called the 'mean' function.
    fn func_inv(x: f64) -> f64;
}

/// Sigmoid activation function.
pub struct Logit;

impl LinkFunc for Logit {
    /// Logit function.
    ///
    /// Returns ln(x / (1 - x)).
    fn func(x: f64) -> f64 {
        (x / (1f64 - x)).ln()
    }

    /// Gradient of logit function.
    ///
    /// Returns 1 / (x * (1-x))
    fn func_grad(x: f64) -> f64 {
        1f64 / (x * (1f64 - x))
    }

    fn func_inv(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

pub struct Bernoulli;

impl Criterion for Bernoulli {
    type Link = Logit;

    fn model_variance(&self, mu: f64) -> f64 {
        1f64 / ((1f64 - mu) * (1f64 - mu))
    }
}
