use std::fmt::Debug;

/// Module for iris dataset.
pub mod iris;
/// Module for trees dataset.
pub mod trees;

/// Dataset container
#[derive(Clone, Debug)]
pub struct Dataset<D, T> where D: Clone + Debug, T: Clone + Debug {

    data: D,
    target: T
}

impl<D, T> Dataset<D, T> where D: Clone + Debug, T: Clone + Debug {

    /// Returns explanatory variable (features)
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Returns objective variable (target)
    pub fn target(&self) -> &T {
        &self.target
    }
}
