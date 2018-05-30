//! The rusty-machine prelude.
//!
//! This module alleviates some common imports used within rusty-machine.

pub use crate::linalg::{Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
pub use crate::linalg::Vector;
pub use crate::linalg::Axes;

pub use crate::learning::SupModel;
pub use crate::learning::UnSupModel;

#[cfg(test)]
mod tests {
    use super::super::prelude::*;

    #[test]
    fn create_mat_from_prelude() {
        let _ = Matrix::new(2, 2, vec![4.0;4]);
    }
}
