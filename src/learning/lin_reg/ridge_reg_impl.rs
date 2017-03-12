use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use linalg::Vector;
use learning::{LearningResult, SupModel};
use learning::error::Error;

use super::RidgeRegressor;

impl Default for RidgeRegressor {
    fn default() -> Self {
        RidgeRegressor {
            alpha: 1.0,
            parameters: None
        }
    }
}

impl RidgeRegressor {

    /// Constructs untrained Ridge regression model.
    ///
    /// Requires L2 regularization parameter (alpha).
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::lin_reg::RidgeRegressor;
    ///
    /// let model = RidgeRegressor::new(1.0);
    /// ```
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0., "alpha must be equal or larger than 0.");
        RidgeRegressor {
            alpha: alpha,
            parameters: None
        }
    }

    /// Get the parameters from the model.
    ///
    /// Returns an option that is None if the model has not been trained.
    pub fn parameters(&self) -> Option<&Vector<f64>> {
        self.parameters.as_ref()
    }
}

impl SupModel<Matrix<f64>, Vector<f64>> for RidgeRegressor {
    /// Train the ridge regression model.
    ///
    /// Takes training data and output values as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::lin_reg::RidgeRegressor;
    /// use rusty_machine::linalg::Matrix;
    /// use rusty_machine::linalg::Vector;
    /// use rusty_machine::learning::SupModel;
    ///
    /// let mut model = RidgeRegressor::default();
    /// let inputs = Matrix::new(3, 1, vec![2.0, 3.0, 4.0]);
    /// let targets = Vector::new(vec![5.0, 10.0, 7.0]);
    ///
    /// model.train(&inputs, &targets).unwrap();
    /// ```
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) -> LearningResult<()> {

        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);
        let xt = full_inputs.transpose();
        // cancel regularization of intercept
        let mut eye = Matrix::<f64>::identity(inputs.cols() + 1);
        unsafe {
            *eye.get_unchecked_mut([0, 0]) = 0.
        }
        let left = &xt * full_inputs + eye * self.alpha;
        let right = &xt * targets;
        self.parameters = Some(left.solve(right).expect("Unable to solve linear equation."));
        Ok(())
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<f64>> {
        if let Some(ref v) = self.parameters {
            let ones = Matrix::<f64>::ones(inputs.rows(), 1);
            let full_inputs = ones.hcat(inputs);
            Ok(full_inputs * v)
        } else {
            Err(Error::new_untrained())
        }
    }
}
