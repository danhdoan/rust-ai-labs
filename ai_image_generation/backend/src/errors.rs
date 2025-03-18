use axum::{http::StatusCode, Json};
use serde::Serialize;
use std::fmt;
use tracing::error;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub enum ErrorCode {
    TextEmbeddingGeneration,
    Inference,
    PostProcessing,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorCode::TextEmbeddingGeneration => write!(f, "Failed to generate Text embedding"),
            ErrorCode::Inference => write!(f, "Failed to run inference"),
            ErrorCode::PostProcessing => write!(f, "Failed to do Post processing"),
        }
    }
}

pub fn handle_error<T: std::fmt::Debug>(
    error_code: ErrorCode,
    err: T,
) -> (StatusCode, Json<ErrorResponse>) {
    error!("{:?}", err);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: format!("{}", error_code),
        }),
    )
}
