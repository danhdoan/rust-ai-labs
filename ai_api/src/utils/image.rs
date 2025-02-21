use tract_ndarray::Array4;
use tract_onnx::prelude::*;

pub fn preprocess_image(image_bytes: Vec<u8>) -> Result<Tensor, String> {
    let img = image::load_from_memory(&image_bytes).map_err(|err| err.to_string())?;

    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    let mut img_arr = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            img_arr[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            img_arr[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            img_arr[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    let tensor = Tensor::from(img_arr);
    Ok(tensor)
}
