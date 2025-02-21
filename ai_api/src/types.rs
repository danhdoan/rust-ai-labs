use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ImageInput {
    pub image: String, // Base64-encoded image string
}

#[derive(Debug, Serialize)]
pub struct ImagePrediction {
    pub label: String,
}
