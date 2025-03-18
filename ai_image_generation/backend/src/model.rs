use candle_core::Device;
use axum::{http::StatusCode, Json};
use lazy_static::lazy_static;
use std::sync::Arc;
use tracing::info;

use crate::ai::stable_diffusion::StableDiffusion;
use crate::errors::{handle_error, ErrorCode, ErrorResponse};
use crate::types::ImagePrompt;

lazy_static! {
    pub static ref MODEL: Arc<StableDiffusion> = {
        let device = Device::cuda_if_available(0)
            .expect("Failed to allocate device");
        let model = StableDiffusion::new(device)
            .expect("Failed to load model");

        Arc::new(model)
    };
}

pub fn run_generation(payload: ImagePrompt)
    -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    info!("{}", format!("Payload: {:?}", payload.prompt));

    let uncond_prompt = "";
    let image_based64 = MODEL.run(&payload.prompt, uncond_prompt)
        .map_err(|err| handle_error(ErrorCode::Inference, err.to_string()))?;

    Ok(image_based64)
}
