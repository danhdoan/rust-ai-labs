use axum::{routing::post, Router};
use tokio::net::TcpListener;
use tracing::Level;

mod errors;
mod routes;
mod model;
mod utils;
mod config;
mod types;

use routes::predict;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let app = Router::new()
        .route("/predict", post(predict));
    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
