use lazy_static::lazy_static;
use tract_onnx::prelude::*;
use crate::config::ONNX_MODEL;

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

lazy_static! {
    pub static ref MODEL: Arc<OnnxModel> = {
        let model = onnx()
            .model_for_path(ONNX_MODEL)
            .expect("Failed to load ONNX model")
            .into_optimized()
            .expect("Failed to optimize model")
            .into_runnable()
            .expect("Failed to make model runnable");

        Arc::new(model)
    };
}

