use crate::errors::ErrorResponse;
use crate::types::ImageResponse;
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
    -> Result<(StatusCode, Json<ImageResponse>), 
                (StatusCode, Json<ErrorResponse>)>
{
    match run_generation(payload) {
        Ok(image) => Ok((StatusCode::OK, Json(ImageResponse {image}))),
        Err(err) => Err(err),
    }
}
