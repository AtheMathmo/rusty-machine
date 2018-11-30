//! The rusty-machine prelude.
//!
//! This module alleviates some common imports used within rusty-machine.

pub use linalg::Axes;
pub use linalg::Vector;
pub use linalg::{BaseMatrix, BaseMatrixMut, Matrix, MatrixSlice, MatrixSliceMut};

pub use learning::SupModel;
pub use learning::UnSupModel;

#[cfg(test)]
mod tests {
    use super::super::prelude::*;

    #[test]
    fn create_mat_from_prelude() {
        let _ = Matrix::new(2, 2, vec![4.0; 4]);
    }
}
