use serde::{Deserialize};

#[derive(Deserialize)]
pub struct ImagePrompt {
    pub prompt: String,
}
