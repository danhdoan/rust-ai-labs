use axum::{http::StatusCode, Json};
use lazy_static::lazy_static;
use tract_onnx::prelude::*;

use crate::config::{IMAGE_CLASS_PATH, ONNX_MODEL_PATH};
use crate::errors::{handle_error, ErrorCode, ErrorResponse};
use crate::utils::classes::load_classes;
use crate::utils::image::preprocess_image;

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

lazy_static! {
    pub static ref MODEL: Arc<OnnxModel> = {
        let model = onnx()
            .model_for_path(ONNX_MODEL_PATH)
            .expect("Failed to load ONNX model")
            .into_optimized()
            .expect("Failed to optimize model")
            .into_runnable()
            .expect("Failed to make model runnable");

        Arc::new(model)
    };
    pub static ref CLASSES: Vec<String> =
        load_classes(IMAGE_CLASS_PATH).expect("Failed to load class file");
}

pub fn run_classification(
    image_bytes: Vec<u8>,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let tensor = preprocess_image(image_bytes)
        .map_err(|err| handle_error(ErrorCode::InvalidInputData, err))?;
    let result = MODEL
        .run(tvec!(tensor.into()))
        .map_err(|err| handle_error(ErrorCode::InferenceFailed, err))?;

    let probs = result[0]
        .to_array_view::<f32>()
        .map_err(|err| handle_error(ErrorCode::OutputConversionFailed, err))?;
    let (max_idx, _) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    if max_idx < CLASSES.len() {
        return Ok(CLASSES[max_idx].clone());
    }
    Err(handle_error(
        ErrorCode::OutputConversionFailed,
        "Class Index Out of Bound",
    ))
}
