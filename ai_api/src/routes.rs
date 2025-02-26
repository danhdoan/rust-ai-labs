use crate::errors::ErrorResponse;
use crate::model::run_classification;
use crate::types::{ImageInput, ImagePrediction};
use axum::{http::StatusCode, Json};
use base64::decode;
use std::time::Instant;
use crate::utils::common::log_elapsed_time;

pub async fn classify(
    Json(payload): Json<ImageInput>,
) -> Result<(StatusCode, Json<ImagePrediction>), (StatusCode, Json<ErrorResponse>)> {
    let start_time = Instant::now();
    let image_bytes = decode(payload.image).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid Base64 input".to_string(),
            }),
        )
    })?;

    if image_bytes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No image uploaded".to_string(),
            }),
        ));
    }

    match run_classification(image_bytes) {
        Ok(label) => {
            log_elapsed_time("Inference", start_time);
            Ok((StatusCode::OK, Json(ImagePrediction { label })))
        }
        Err(err) => Err(err),
    }
}
