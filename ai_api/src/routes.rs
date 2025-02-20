use crate::errors::{handle_error, ErrorCode, ErrorResponse};
use crate::model::{run_batch_inference, run_sample_inference};
use crate::types::{BatchPrediction, InputBatchData, InputData, Prediction};
use axum::{http::StatusCode, Json};
use std::time::Instant;
use tracing::info;

pub async fn predict(
    Json(payload): Json<InputData>,
) -> Result<(StatusCode, Json<Prediction>), (StatusCode, Json<ErrorResponse>)> {
    info!("Payload: {:?}", &payload.features);

    let start_time = Instant::now();
    if payload.features.len() != 5 {
        return Err(handle_error(
            ErrorCode::InvalidInputData,
            "Invalid input shape",
        ));
    }

    match run_sample_inference(payload) {
        Ok(result) => {
            let elapsed_time = start_time.elapsed();
            info!("Inference completed in: {:.3?}", elapsed_time);
            Ok((StatusCode::OK, Json(Prediction { result })))
        }
        Err(err) => Err(err),
    }
}

pub async fn batch_predict(
    Json(payload): Json<InputBatchData>,
) -> Result<(StatusCode, Json<BatchPrediction>), (StatusCode, Json<ErrorResponse>)> {
    info!("Payload: {:?}", &payload.features);

    let start_time = Instant::now();
    if payload.features.is_empty() {
        return Err(handle_error(
            ErrorCode::InvalidInputData,
            "Empty input for prediction",
        ));
    }
    if payload.features[0].len() != 5 {
        return Err(handle_error(
            ErrorCode::InvalidInputData,
            "Invalid input shape",
        ));
    }

    match run_batch_inference(payload) {
        Ok(results) => {
            let elapsed_time = start_time.elapsed();
            info!("Inference completed in: {:.3?}", elapsed_time);
            Ok((StatusCode::OK, Json(BatchPrediction { results })))
        }
        Err(err) => Err(err),
    }
}
