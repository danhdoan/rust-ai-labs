use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct InputData {
    pub features: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct InputBatchData {
    pub features: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub result: f32,
}

#[derive(Debug, Serialize)]
pub struct BatchPrediction {
    pub results: Vec<f32>,
}
