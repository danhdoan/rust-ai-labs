use axum::{http::StatusCode, Json};
use lazy_static::lazy_static;
use std::sync::Arc;
use tch::{CModule, Device, Tensor};

use crate::config::{IMAGE_CLASS_PATH, PYTORCH_MODEL_PATH};
use crate::errors::{handle_error, ErrorCode, ErrorResponse};
use crate::utils::classes::load_classes;
use crate::utils::image::preprocess_image;

lazy_static! {
    pub static ref MODEL: Arc<CModule> = {
        let model = CModule::load_on_device(PYTORCH_MODEL_PATH, Device::Cuda(0))
            .expect("Failed to load Torch model");

        Arc::new(model)
    };
    pub static ref CLASSES: Vec<String> =
        load_classes(IMAGE_CLASS_PATH).expect("Failed to load class file");
}

pub fn perform_inference(tensor: Tensor) -> Result<String, (ErrorCode, String)> {
    let device = Device::Cuda(0);
    let tensor = tensor.to_device(device);
    let _guard = tch::no_grad_guard();

    let output = MODEL
        .forward_ts(&[tensor])
        .map_err(|err| (ErrorCode::InferenceFailed, err.to_string()))?;

    let probs = output.softmax(-1, tch::Kind::Float);
    let max_idx = probs.argmax(-1, false).int64_value(&[]);
    drop(output);
    drop(probs);
    if (max_idx as usize) < CLASSES.len() {
        return Ok(CLASSES[max_idx as usize].clone());
    }
    Err((
        ErrorCode::OutputConversionFailed,
        "Class Index Out of Bound".to_string(),
    ))
}
pub fn run_classification(
    image_bytes: Vec<u8>,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let tensor = preprocess_image(image_bytes)
        .map_err(|err| handle_error(ErrorCode::InvalidInputData, err))?;

    let output =
        perform_inference(tensor).map_err(|(err_code, err_msg)| handle_error(err_code, err_msg))?;
    Ok(output)
}
