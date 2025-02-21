use axum::{routing::post, Router};
use tokio::net::TcpListener;
use tracing::{info, Level};

mod config;
mod errors;
mod model;
mod routes;
mod types;
mod utils;

use routes::classify;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let app = Router::new().route("/classify", post(classify));
    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();

    info!("AI API server ready!");
    axum::serve(listener, app).await.unwrap();
}
