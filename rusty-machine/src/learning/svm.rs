//! Support Vector Machine Module
//!
//! Contains implementation of Support Vector Machine
//! using the [Pegasos training algorithm](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf).
//!
//! The SVM model currently only support binary classification.
//! The model inputs should be a matrix and the training targets are
//! in the form of a vector of `-1`s and `1`s.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::svm::SVM;
//! use rusty_machine::learning::SupModel;
//!
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::linalg::vector::Vector;
//!
//! let inputs = Matrix::new(4,1,vec![1.0,3.0,5.0,7.0]);
//! let targets = Vector::new(vec![-1.,-1.,1.,1.]);
//!
//! let mut svm_mod = SVM::default();
//!
//! // Train the model
//! svm_mod.train(&inputs, &targets);
//! 
//! // Now we'll predict a new point
//! let new_point = Matrix::new(1,1,vec![10.]);
//! let output = svm_mod.predict(&new_point);
//!
//! // Hopefully we classified our new point correctly!
//! assert!(output[0] == 1f64, "Our classifier isn't very good!");
//! ```


use linalg::matrix::Matrix;
use linalg::vector::Vector;

use learning::toolkit::kernel::{Kernel, SquaredExp};
use learning::SupModel;

use rand;
use rand::Rng;

/// Support Vector Machine
pub struct SVM<K: Kernel> {
    ker: K,
    alpha: Option<Vector<f64>>,
    train_inputs: Option<Matrix<f64>>,
    train_targets: Option<Vector<f64>>,
    lambda: f64,
}

impl Default for SVM<SquaredExp> {
    fn default() -> SVM<SquaredExp> {
        SVM {
            ker: SquaredExp::default(),
            alpha: None,
            train_inputs: None,
            train_targets: None,
            lambda: 0.3f64,
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
		}
	}
}

impl<K: Kernel> SVM<K> {

    /// Construct a kernel matrix
    fn ker_mat(&self, m1: &Matrix<f64>, m2: &Matrix<f64>) -> Matrix<f64> {
        assert_eq!(m1.cols(), m2.cols());
        let cols = m1.cols();

        let dim1 = m1.rows();
        let dim2 = m2.rows();

        let mut ker_data = Vec::with_capacity(dim1 * dim2);

        for i in 0..dim1 {
            for j in 0..dim2 {
                ker_data.push(self.ker.kernel(&m1.data()[i * cols..(i + 1) * cols],
                                              &m2.data()[j * cols..(j + 1) * cols]));
            }
        }

        Matrix::new(dim1, dim2, ker_data)
    }
}

impl<K: Kernel> SupModel<Matrix<f64>, Vector<f64>> for SVM<K> {
    fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
    	let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);

        if let (&Some(ref alpha),
                &Some(ref train_inputs),
                &Some(ref train_targets)) = (&self.alpha, &self.train_inputs, &self.train_targets) {
            let ker_mat = self.ker_mat(&full_inputs, train_inputs);
            let weight_vec = alpha.elemul(train_targets) / self.lambda;

            let plane_dist = ker_mat * weight_vec;

            plane_dist.apply(&|d| d.signum())
        }
        else {
        	panic!("Model has not been trained.");
        }
    }

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        let n = inputs.rows();

        let mut rng = rand::thread_rng();

        let mut alpha = vec![0f64; n];

        let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        let full_inputs = ones.hcat(inputs);
        let m = full_inputs.cols();

        // TODO: Make T a variable instead of 1000 constant.
        for t in 0..1000 {
            let i = rng.gen_range(0, n);
            let mut sum = 0f64;
            for j in 0..n {
                sum += alpha[j] * targets[j] *
                       self.ker.kernel(&full_inputs.data()[i * m..(i + 1) * m],
                                       &full_inputs.data()[j * m..(j + 1) * m]);
            }
            sum *= targets[i] / (self.lambda * (t as f64));

            if sum < 1f64 {
                alpha[i] = alpha[i] + 1f64;
            }
        }

        self.alpha = Some(Vector::new(alpha) / 1000f64);
        self.train_inputs = Some(full_inputs);
        self.train_targets = Some(targets.clone());
    }
}
