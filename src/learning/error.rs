//! Error handling for the learning module.

use std::boxed::Box;
use std::convert::Into;
use std::error;
use std::fmt;
use std::marker::{Send, Sync};

use rulinalg;

/// An error related to the learning module.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    error: Box<error::Error + Send + Sync>,
}

/// Types of errors produced in the learning module.
///
/// List intended to grow and so you should
/// be wary of matching against explicitly.
#[derive(Debug)]
pub enum ErrorKind {
    /// The parameters used to define the model are not valid.
    InvalidParameters,
    /// The input data to the model is not valid.
    InvalidData,
    /// The action could not be carried out as the model was in an invalid state.
    InvalidState,
    /// The model has not been trained
    UntrainedModel,
    /// Linear algebra related error
    LinearAlgebra
}

impl Error {
    /// Construct a new `Error` of a particular `ErrorKind`.
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
        where E: Into<Box<error::Error + Send + Sync>>
    {
        Error {
            kind: kind,
            error: error.into(),
        }
    }

    /// Returns a new error for an untrained model
    ///
    /// This function is unstable and may be removed with changes to the API.
    pub fn new_untrained() -> Error {
        Error::new(ErrorKind::UntrainedModel, "The model has not been trained.")
    }

    /// Get the kind of this `Error`.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl From<rulinalg::error::Error> for Error {
    fn from(e: rulinalg::error::Error) -> Error {
        Error::new(ErrorKind::LinearAlgebra, <rulinalg::error::Error as error::Error>::description(&e))
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.error.description()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.error.fmt(f)
    }
}
