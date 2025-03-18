use anyhow::Result;
use candle_core::{DType, Device};
use candle_transformers::models::stable_diffusion::{vae::AutoEncoderKL, StableDiffusionConfig};

use crate::ai::model_files::ModelFile;

pub fn build_vae_model(
    vae_weight_path: &str,
    sd_config: &StableDiffusionConfig,
    device: &Device,
) -> Result<AutoEncoderKL> {
    let vae_weights = ModelFile::Vae.get(vae_weight_path.to_string())?;
    let vae_model = sd_config.build_vae(vae_weights, device, DType::F16)?;
    Ok(vae_model)
}
