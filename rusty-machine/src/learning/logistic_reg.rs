use learning::SupModel;
use linalg::matrix::Matrix;
use linalg::vector::Vector;
use learning::toolkit::activ_fn::ActivationFunc;
use learning::toolkit::activ_fn::Sigmoid;
use learning::toolkit::cost_fn::CostFunc;
use learning::toolkit::cost_fn::CrossEntropyError;
use learning::optim::grad_desc::GradientDesc;
use learning::optim::OptimAlgorithm;
use learning::optim::Optimizable;

/// Logistic Regression Model.
///
/// Contains option for optimized parameter.
pub struct LogisticRegressor {
    /// The parameters for the regression model.
    parameters: Option<Vector<f64>>,
    gd: GradientDesc,
}

impl Default for LogisticRegressor {
    fn default() -> LogisticRegressor {
        LogisticRegressor { parameters: None, gd: GradientDesc::default() }
    }
}
impl LogisticRegressor {
    /// Constructs untrained logistic regression model.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::logistic_reg::LogisticRegressor;
    /// use rusty_machine::learning::optim::grad_desc::GradientDesc;
    ///
    /// let gd = GradientDesc::default();
    /// let mut logistic_mod = LogisticRegressor::new(gd);
    /// ```
    pub fn new(gd: GradientDesc) -> LogisticRegressor {
        LogisticRegressor { parameters: None, gd: gd }
    }

    /// Get the parameters from the model.
    ///
    /// Returns an option that is None if the model has not been trained.
    pub fn parameters(&self) -> Option<Vector<f64>> {
        match self.parameters {
            None => None,
            Some(ref x) => Some(x.clone()),
        }
    }
}

impl SupModel<Matrix<f64>, Vector<f64>> for LogisticRegressor {
    /// Train the logistic regression model.
    ///
    /// Takes training data and output values as input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::logistic_reg::LogisticRegressor;
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::vector::Vector;
    /// use rusty_machine::learning::SupModel;
    ///
    /// let mut logistic_mod = LogisticRegressor::default();
    /// let inputs = Matrix::new(3,2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
    /// let targets = Vector::new(vec![5.0, 6.0, 7.0]);
    ///
    /// logistic_mod.train(&inputs, &targets);
    /// ```
    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let initial_params = vec![0.5; inputs.cols()];

        let optimal_w = self.gd.optimize(self, &initial_params[..], inputs, targets);
        self.parameters = Some(Vector::new(optimal_w));
    }

    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
        match self.parameters {
            Some(ref v) => (inputs * v).apply(&Sigmoid::func),
            None => panic!("Model has not been trained."),
        }
    }
}

impl Optimizable for LogisticRegressor {
    type Inputs = Matrix<f64>;
    type Targets = Vector<f64>;

    fn compute_grad(&self, params: &[f64], inputs: &Matrix<f64>, targets: &Vector<f64>) -> (f64, Vec<f64>) {
        
        let beta_vec = Vector::new(params.to_vec());
        let outputs = (inputs * beta_vec).apply(&Sigmoid::func);

        let cost = CrossEntropyError::cost(&outputs, targets);
        let grad = (inputs.transpose() * (outputs-targets)) / (inputs.rows() as f64);

        println!("Cost is {0}", cost);

        (cost, grad.into_vec())
    }

}