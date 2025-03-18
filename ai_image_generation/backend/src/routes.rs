use crate::errors::ErrorResponse;
use crate::types::ImagePrompt;
use axum::{http::StatusCode, Json};
use crate::model::run_generation;

pub async fn health_check()
    -> Result<(StatusCode, String), 
                (StatusCode, Json<ErrorResponse>)>
{
    Ok((StatusCode::OK, "Server is working!".to_string()))
}

pub async fn generate(Json(payload): Json<ImagePrompt>) 
    -> Result<(StatusCode, String), 
                (StatusCode, Json<ErrorResponse>)>
{
    match run_generation(payload) {
        Ok(msg) => Ok((StatusCode::OK, msg)),
        Err(err) => Err(err),
    }
}
