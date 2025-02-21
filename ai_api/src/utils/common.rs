use std::time::Instant;
use tracing::info;

pub fn log_elapsed_time(label: &str, start_time: Instant) {
    let elapsed_time = start_time.elapsed();
    info!("⏱️ {} took {:.3?} seconds", label, elapsed_time);
}
