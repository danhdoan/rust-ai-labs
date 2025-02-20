use axum::{http::StatusCode, Json};
use lazy_static::lazy_static;
use tract_ndarray::ArrayD;
use tract_onnx::prelude::*;

use crate::config::ONNX_MODEL;
use crate::errors::{handle_error, ErrorCode, ErrorResponse};
use crate::types::{InputBatchData, InputData};

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

lazy_static! {
    pub static ref MODEL: Arc<OnnxModel> = {
        let model = onnx()
            .model_for_path(ONNX_MODEL)
            .expect("Failed to load ONNX model")
            .into_optimized()
            .expect("Failed to optimize model")
            .into_runnable()
            .expect("Failed to make model runnable");

        Arc::new(model)
    };
}

pub fn run_sample_inference(payload: InputData) -> Result<f32, (StatusCode, Json<ErrorResponse>)> {
    let input_tensor = ArrayD::<f32>::from_shape_vec(vec![1, 5], payload.features)
        .map_err(|err| handle_error(ErrorCode::InvalidInputData, err))?;
    let input_tract = Tensor::from(input_tensor);
    let output = MODEL
        .run(tvec!(input_tract.into()))
        .map_err(|err| handle_error(ErrorCode::InferenceFailed, err))?;

    let result = *output[0].to_scalar::<f32>().unwrap();
    Ok(result)
}

pub fn run_batch_inference(
    payload: InputBatchData,
) -> Result<Vec<f32>, (StatusCode, Json<ErrorResponse>)> {
    let batch_size = payload.features.len();
    let num_features = payload.features[0].len();

    let flat_input: Vec<f32> = payload.features.into_iter().flatten().collect();
    let input_tensor = ArrayD::<f32>::from_shape_vec(vec![batch_size, num_features], flat_input)
        .map_err(|err| handle_error(ErrorCode::InvalidInputData, err))?;

    let input_tract = Tensor::from(input_tensor);

    let output = MODEL
        .run(tvec!(input_tract.into()))
        .map_err(|err| handle_error(ErrorCode::InferenceFailed, err))?;

    let results: Vec<f32> = output[0]
        .to_array_view::<f32>()
        .map_err(|err| handle_error(ErrorCode::OutputConversionFailed, err))?
        .iter()
        .cloned()
        .collect();
    Ok(results)
}
