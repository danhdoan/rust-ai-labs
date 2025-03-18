use base64::{engine::general_purpose, Engine};
use candle_core::{Tensor, DType, Device };
use anyhow::{bail, Result};
use image::{self, ImageReader, ImageBuffer};
use std::io::Cursor;

pub fn _image_preprocess<T: AsRef<std::path::Path>>(path: T) -> Result<Tensor> {
    let img = ImageReader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

pub fn tensor_to_image(img: &Tensor)
-> Result<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        bail!("Invalid number of channel for input");
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
        .expect("Failed to convert Tensor to image");
    Ok(image)
}

pub fn image_to_base64(image_buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>)
-> Result<String> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);

    image_buffer.write_to(&mut cursor, image::ImageFormat::Png)
        .expect("Failed to write image to buffer");

    Ok(general_purpose::STANDARD.encode(&buffer))
}
