use std::time::Instant;
use tracing::info;
use axum::{Json, http::StatusCode};
use tract_onnx::prelude::*;
use tract_ndarray::ArrayD;
use crate::model::MODEL;
use crate::types::{InputData, Prediction};
use crate::errors::{ErrorCode, ErrorResponse, handle_error};

pub async fn predict(Json(payload): Json<InputData>) 
                                -> Result<(StatusCode, Json<Prediction>), (StatusCode, Json<ErrorResponse>)> {
    info!("Payload: {:?}", &payload.features);

    let start_time = Instant::now();
    if payload.features.len() != 5 {
        return Err(handle_error(ErrorCode::InvalidInputData, "Invalid input shape"));
    }

    let input_tensor = ArrayD::<f32>::from_shape_vec(vec![1, 5], payload.features)
        .map_err(|err| handle_error(ErrorCode::InvalidInputData, err))?;
    let input_tract = Tensor::from(input_tensor);
    let output = MODEL.run(tvec!(input_tract.into()))
        .map_err(|err| handle_error(ErrorCode::InferenceFailed, err))?;

    let result = *output[0].to_scalar::<f32>().unwrap();
    let elapsed_time = start_time.elapsed();
    info!("Inference completed in: {:.3?}", elapsed_time);
    Ok((StatusCode::OK, Json(Prediction { result })))
}
