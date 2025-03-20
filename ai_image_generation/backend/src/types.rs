use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ImagePrompt {
    pub prompt: String,
    pub neg_prompt: String,
}

#[derive(Serialize)]
pub struct ImageResponse {
    pub image: String,
}
