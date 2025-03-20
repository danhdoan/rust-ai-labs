use anyhow::Result;
use candle_core::{DType, Device};
use candle_transformers::models::stable_diffusion::{unet_2d::UNet2DConditionModel, StableDiffusionConfig};

use crate::ai::model_files::ModelFile;

pub fn build_unet_model(
    unet_weight_path: &str,
    sd_config: &StableDiffusionConfig,
    device: &Device,
) -> Result<UNet2DConditionModel> {
    let unet_weights = ModelFile::Unet.get(unet_weight_path.to_string())?;
    let in_channels = 4;
    let use_flash_attn = true;
    let unet = sd_config.build_unet(unet_weights, device, in_channels, use_flash_attn, DType::F16)?;

    Ok(unet)
}
