pub mod minmax;
pub mod standardize;

use learning::error;

pub use self::minmax::MinMaxScaler;
pub use self::standardize::Standardizer;

/// Trait for data transformers
pub trait Transformer<T> {
    /// Transforms the inputs and stores the transformation in the Transformer
    fn transform(&mut self, inputs: T) -> Result<T, error::Error>;
}