use anyhow::Result;
use candle_transformers::models::stable_diffusion::{self, clip};
use candle_core::{DType, Device};

use crate::ai::model_files::ModelFile;

pub fn build_text_encoder(
    clip_weight_path: &str, 
    clip_config: &clip::Config,
    device: &Device,
) -> Result<clip::ClipTextTransformer> {
    let clip_weights = ModelFile::Clip.get(clip_weight_path.to_string())?;
    let text_encoder_model = stable_diffusion::build_clip_transformer(
        clip_config, clip_weights, device, DType::F16)?;

    Ok(text_encoder_model)
}

