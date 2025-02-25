use image::io::Reader as ImageReader;
use std::io::Cursor;
use tch::{Kind, Tensor};

pub fn preprocess_image(image_bytes: Vec<u8>) -> Result<Tensor, String> {
    let cursor = Cursor::new(image_bytes);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    let mut tensor = Tensor::from_slice(img.as_raw())
        .reshape([height as i64, width as i64, 3])
        .permute([2, 0, 1])
        .to_kind(Kind::Float)
        / 255.0;
    tensor = tensor.unsqueeze(0);
    Ok(tensor)
}
