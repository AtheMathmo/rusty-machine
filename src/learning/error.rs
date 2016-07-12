//! Error handling for the learning module.

use std::boxed::Box;
use std::convert::Into;
use std::error;
use std::fmt;
use std::marker::{Send, Sync};

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

    /// Get the kind of this `Error`.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
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
