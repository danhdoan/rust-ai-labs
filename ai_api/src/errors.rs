use std::fmt;
use serde::{Serialize};
use axum::{Json, http::StatusCode};
use tracing::error;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub enum ErrorCode {
    InvalidInputData,
    InferenceFailed
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorCode::InvalidInputData => write!(f, "Invalid input for inference"),
            ErrorCode::InferenceFailed => write!(f, "Failed to run inference"),
        }
    }
}

pub fn handle_error<T: std::fmt::Debug>(error_code: ErrorCode, err: T) 
-> (StatusCode, Json<ErrorResponse>) {
    error!("{:?}", err);
    (StatusCode::INTERNAL_SERVER_ERROR, 
     Json(ErrorResponse {
         error: format!("{}", error_code)
     })
    )
}

