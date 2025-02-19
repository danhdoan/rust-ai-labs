use serde::{Serialize,  Deserialize};

#[derive(Debug, Deserialize)]
pub struct InputData {
    pub features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub result: f32,
}

