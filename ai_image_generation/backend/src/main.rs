use axum::{routing::{post, get}, Router};
use tokio::net::TcpListener;
use tracing::{info, Level};
use tower_http::services::ServeDir;

mod image_lib;
mod configs;
mod errors;
mod ai;
mod model;
mod routes;
mod types;
mod utils;

use routes::{health_check, generate};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/generate", post(generate))
        .fallback_service(ServeDir::new("public"));

    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();

    info!("AI ImageGen server ready!");
    axum::serve(listener, app).await.unwrap();
}
