use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_classes(path: &str) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let reader = BufReader::new(file);
    let classes: Vec<String> = reader
        .lines()
        .map(|line| line.unwrap_or_else(|_| "Unknown".to_string()))
        .collect();

    Ok(classes)
}
