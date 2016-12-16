//! Principal Component Analysis Module
//!
//! Contains implementation of PCA.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::pca::PCA;
//! use rusty_machine::learning::UnSupModel;
//!
//! use rusty_machine::linalg::Matrix;
//! let mut pca = PCA::default();
//!
//! let inputs = Matrix::new(3, 2, vec![1., 0.1,
//!                                     3., 0.2,
//!                                     4., 0.2]);
//! // Train the model
//! pca.train(&inputs).unwrap();
//!
//! // Now we'll predict a new point
//! let new_data = Matrix::new(1, 2, vec![2., 0.1]);
//! let output = pca.predict(&new_data).unwrap();
//!
//! assert_eq!(output[[0, 0]], -0.6686215718235227);
//! ```

use linalg::{Matrix, BaseMatrix, Axes};
use linalg::Vector;

use learning::{LearningResult, UnSupModel};
use learning::error::Error;

/// Principal Component Analysis
#[derive(Debug)]
pub struct PCA {
    // Flag whether to centering inputs
    center: bool,

    // Center of inputs
    centers: Option<Vector<f64>>,
    // Principal components
    components: Option<Matrix<f64>>
}

impl PCA {

    /// Constructs untrained PCA model.
    ///
    /// Requires flag whether centering inputs to be specified.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::pca::PCA;
    ///
    /// let model = PCA::new(true);
    /// ```
    pub fn new(center: bool) -> PCA {
        PCA {
            center: center,

            centers: None,
            components: None
        }
    }

    /// Returns principal components (matrix which contains eigenvectors as columns)
    pub fn components(&self) -> LearningResult<&Matrix<f64>>  {
        match self.components {
            None => Err(Error::new_untrained()),
            Some(ref rot) => { Ok(rot) }
        }
    }
}

/// The default PCA.
///
/// The default is:
///
/// - `center` = `true`
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::pca::PCA;
///
/// let model = PCA::default();
/// ```
impl Default for PCA {
    fn default() -> Self {
        PCA {
            center: true,

            centers: None,
            components: None
        }
    }
}

/// Train the model and predict the model output from new data.
impl UnSupModel<Matrix<f64>, Matrix<f64>> for PCA {

    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>>  {

        match self.components {
            None => Err(Error::new_untrained()),
            Some(ref comp) => {

                if self.center == true {
                    match self.centers {
                        None => return Err(Error::new_untrained()),
                        Some(ref centers) => {
                            let data = centering(inputs, &centers);
                            Ok(data * comp)
                        }
                    }
                } else {
                    Ok(inputs * comp)
                }
            }
        }
    }

    fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        let data = if self.center == true {
            let centers = inputs.mean(Axes::Row);
            let m = centering(inputs, &centers);
            self.centers = Some(centers);
            m
        } else {
            inputs.clone()
        };
        let (_, _, v) = data.svd().unwrap();
        self.components = Some(v);
        Ok(())
    }
}

/// Subtract center Vector from each rows
fn centering(inputs: &Matrix<f64>, centers: &Vector<f64>) -> Matrix<f64> {
    unsafe {
        Matrix::from_fn(inputs.rows(), inputs.cols(),
                    |c, r| inputs.get_unchecked([r, c]) - centers.data().get_unchecked(c))
    }
}

#[test]
fn test_centering() {
    let m = Matrix::new(2, 3, vec![1., 2., 3.,
                                   2., 4., 4.]);
    let centers = m.mean(Axes::Row);
    assert_vector_eq!(centers, Vector::new(vec![1.5, 3., 3.5]), comp=abs, tol=1e-8);
    let exp = Matrix::new(2, 3, vec![-0.5, -1., -0.5,
                                     0.5, 1., 0.5]);
    assert_matrix_eq!(centering(&m, &centers), exp, comp=abs, tol=1e-8);
}