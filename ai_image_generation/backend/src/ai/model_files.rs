use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFile {
    Tokenizer,
    Clip,
    Unet,
    Vae,
}

impl ModelFile {
    pub fn get(
        &self,
        filename: String,
    ) -> Result<std::path::PathBuf> {
     Ok(std::path::PathBuf::from(filename))
    }
}
