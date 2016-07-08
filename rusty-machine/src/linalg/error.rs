//! Error handling for the linalg module.

use std::boxed::Box;
use std::convert::Into;
use std::error;
use std::fmt;
use std::marker::{Send, Sync};

/// An error related to the linalg module.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    error: Box<error::Error + Send + Sync>,
}

/// Types of errors produced in the linalg module.
///
/// List intended to grow and so you should
/// be wary of matching against explicitly.
#[derive(Debug)]
pub enum ErrorKind {
    /// An argument did not uphold a necessary criteria for the function.
    InvalidArg,
    /// A failure to decompose due to some property of the data.
    DecompFailure,
    /// A failure due to some algebraic constraints not being met.
    AlgebraFailure,
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
