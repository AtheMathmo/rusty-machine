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
            self.criterion.apply_link_inv(full_inputs * v)
        } else {
            panic!("The model has not been trained.");
        }
    }

    /// Train the model using inputs and targets.
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let n = inputs.rows();

        assert!(n == targets.size(),
                "Training data do not have the same dimensions.");

        // Construct mu as a non-zero estimate of targets.
        let mut mu_data = Vec::with_capacity(n);
        for y in targets.data() {
        	mu_data.push(0.5);//if *y > 0f64 { *y } else { 1e-3 })
        }

        let mut mu = Vector::new(mu_data);
        let mut z = mu.clone();
        let mut beta : Vector<f64> = Vector::new(vec![]);

        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);
        let x_t = full_inputs.transpose();

        // Iterate to convergence
        for k in 0..15 {
            let mut w_diag = Vec::with_capacity(n);
            let mut y_bar_data = Vec::with_capacity(n);

            for (idx, m) in mu.data().iter().enumerate() {
                y_bar_data.push(self.criterion.compute_y_bar(targets[idx], *m));
                w_diag.push(self.criterion.compute_working_weight(*m));
            }

            let i = 5;

            println!("-------- Iteration {0} --------", k+1);
            println!("mu[{0}] = {1}", i, mu[i]);
            println!("y[{0}] = {1}", i, targets[i]);
            println!("y_bar[{0}] = {1}", i, y_bar_data[i]);
            println!("w[{0}] = {1}", i, w_diag[i]);

            let w = Matrix::from_diag(&w_diag);
            let y_bar = Vector::new(y_bar_data);

            let x_t_w = &x_t * w;
            beta = (&x_t_w * &full_inputs).inverse() * x_t_w * z;

            //println!("Beta = {:?}", beta.data());

            // Update z and mu
            let fitted = &full_inputs * &beta;
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

    fn compute_working_weight(&self, mu: f64) -> f64 {
        1f64 / (self.model_variance(mu) * self.link_grad(mu) * self.link_grad(mu))
    }

    fn compute_y_bar(&self, y:f64, mu: f64) -> f64 {
        self.link_grad(mu) * (y - mu)
    }

    fn apply_link_func(&self, vec: Vector<f64>) -> Vector<f64> {
        vec.apply(&Self::Link::func)
    }

    fn apply_link_inv(&self, vec: Vector<f64>) -> Vector<f64> {
    	vec.apply(&Self::Link::func_inv)
    }

    fn link_grad(&self, mu: f64) -> f64 {
        Self::Link::func_grad(mu)
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
        mu * (1f64 - mu)
    }

    fn compute_working_weight(&self, mu: f64) -> f64 {
        self.model_variance(mu)
    }

    fn compute_y_bar(&self, y:f64, mu: f64) -> f64 {
        let target_diff = y - mu;

        if target_diff.abs() < 1e-15 {
            0f64
        }
        else {
            self.link_grad(mu) * target_diff
        }
    }
}
