//! Support Vector Machine Module
//!
//! Contains implementation of Support Vector Machine using the
//! [Pegasos training algorithm](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf).
//!
//! The SVM models currently only support binary classification.
//! The model inputs should be a matrix and the training targets are
//! in the form of a vector of `-1`s and `1`s.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::svm::SVM;
//! use rusty_machine::learning::SupModel;
//!
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//!
//! let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
//! let targets = Vector::new(vec![-1.,-1.,1.,1.]);
//!
//! let mut svm_mod = SVM::default();
//!
//! // Train the model
//! svm_mod.train(&inputs, &targets).unwrap();
//!
//! // Now we'll predict a new point
//! let new_point = Matrix::new(1,1,vec![10.]);
//! let output = svm_mod.predict(&new_point).unwrap();
//!
//! // Hopefully we classified our new point correctly!
//! assert!(output[0] == 1f64, "Our classifier isn't very good!");
//! ```


use linalg::{Matrix, BaseMatrix};
use linalg::Vector;

use learning::toolkit::kernel::{Kernel, SquaredExp};
use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};

use rand;
use rand::Rng;

/// Support Vector Machine
#[derive(Debug)]
pub struct SVM<K: Kernel> {
    ker: K,
    alpha: Option<Vector<f64>>,
    train_inputs: Option<Matrix<f64>>,
    train_targets: Option<Vector<f64>>,
    lambda: f64,
    /// Number of iterations for training.
    pub optim_iters: usize,
}

/// The default Support Vector Machine.
///
/// The defaults are:
///
/// - `ker` = `SquaredExp::default()`
/// - `lambda` = `0.3`
/// - `optim_iters` = `100`
impl Default for SVM<SquaredExp> {
    fn default() -> SVM<SquaredExp> {
        SVM {
            ker: SquaredExp::default(),
            alpha: None,
            train_inputs: None,
            train_targets: None,
            lambda: 0.3f64,
            optim_iters: 100,
        }
    }
}

impl<K: Kernel> SVM<K> {
    /// Constructs an untrained SVM with specified
    /// kernel and lambda which determins the hardness
    /// of the margin.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::svm::SVM;
    /// use rusty_machine::learning::toolkit::kernel::SquaredExp;
    ///
    /// let _ = SVM::new(SquaredExp::default(), 0.3);
    /// ```
    pub fn new(ker: K, lambda: f64) -> SVM<K> {
        SVM {
            ker: ker,
            alpha: None,
            train_inputs: None,
            train_targets: None,
            lambda: lambda,
            optim_iters: 100,
        }
    }
}

impl<K: Kernel> SVM<K> {
    /// Construct a kernel matrix
    fn ker_mat(&self, m1: &Matrix<f64>, m2: &Matrix<f64>) -> LearningResult<Matrix<f64>> {
        if m1.cols() != m2.cols() {
            Err(Error::new(ErrorKind::InvalidState,
                           "Inputs to kernel matrices have different column counts."))
        } else {
            let dim1 = m1.rows();
            let dim2 = m2.rows();

            let mut ker_data = Vec::with_capacity(dim1 * dim2);
            ker_data.extend(m1.row_iter().flat_map(|row1| {
                m2.row_iter()
                    .map(move |row2| self.ker.kernel(row1.raw_slice(), row2.raw_slice()))
            }));

            Ok(Matrix::new(dim1, dim2, ker_data))
        }
    }
}

/// Train the model using the Pegasos algorithm and
/// predict the model output from new data.
impl<K: Kernel> SupModel<Matrix<f64>, Vector<f64>> for SVM<K> {
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<f64>> {
        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        if let (&Some(ref alpha), &Some(ref train_inputs), &Some(ref train_targets)) =
               (&self.alpha, &self.train_inputs, &self.train_targets) {
            let ker_mat = self.ker_mat(&full_inputs, train_inputs)?;
            let weight_vec = alpha.elemul(train_targets) / self.lambda;

            let plane_dist = ker_mat * weight_vec;

            Ok(plane_dist.apply(&|d| d.signum()))
        } else {
            Err(Error::new_untrained())
        }
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) -> LearningResult<()> {
        let n = inputs.rows();

        let mut rng = rand::thread_rng();

        let mut alpha = vec![0f64; n];

        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        for t in 0..self.optim_iters {
            let i = rng.gen_range(0, n);
            let row_i = full_inputs.select_rows(&[i]);
            let sum = full_inputs.row_iter()
                .fold(0f64, |sum, row| sum + self.ker.kernel(row_i.data(), row.raw_slice())) *
                      targets[i] / (self.lambda * (t as f64));

            if sum < 1f64 {
                alpha[i] += 1f64;
            }
        }

        self.alpha = Some(Vector::new(alpha) / (self.optim_iters as f64));
        self.train_inputs = Some(full_inputs);
        self.train_targets = Some(targets.clone());

        Ok(())
    }
}
