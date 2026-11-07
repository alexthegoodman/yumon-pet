//! Diagnostics shared by every stage. Source errors carry the 1-based line of
//! the statement and its text so messages can name both.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct WirError {
    /// 1-based line of the statement, if the error comes from source.
    pub line: Option<usize>,
    /// The statement text (comments stripped, continuation lines joined).
    pub statement: Option<String>,
    pub message: String,
}

impl WirError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { line: None, statement: None, message: message.into() }
    }

    pub fn at(line: usize, statement: &str, message: impl Into<String>) -> Self {
        Self { line: Some(line), statement: Some(statement.to_string()), message: message.into() }
    }
}

impl fmt::Display for WirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.line, &self.statement) {
            (Some(line), Some(stmt)) => write!(f, "line {line}: `{stmt}`: {}", self.message),
            (Some(line), None) => write!(f, "line {line}: {}", self.message),
            _ => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for WirError {}

pub type Result<T> = std::result::Result<T, WirError>;

/// A non-fatal diagnostic (unused crease, rank-deficient solve, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct Warning {
    pub line: Option<usize>,
    pub message: String,
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "warning: line {line}: {}", self.message),
            None => write!(f, "warning: {}", self.message),
        }
    }
}
