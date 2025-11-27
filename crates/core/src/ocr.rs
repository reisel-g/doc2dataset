use image::DynamicImage;

use crate::error::{DcfError, Result};

#[cfg(feature = "ocr")]
use {
    image::{codecs::png::PngEncoder, ColorType, ImageEncoder},
    leptess::LepTess,
    std::io::Write,
    tempfile::NamedTempFile,
};

#[cfg(feature = "ocr")]
pub fn image_to_text(image: &DynamicImage, languages: &[String]) -> Result<String> {
    let lang = if languages.is_empty() {
        "eng".to_string()
    } else {
        languages.join("+")
    };
    let mut tess = LepTess::new(None, &lang)
        .map_err(|e| DcfError::Other(format!("failed to initialise tesseract: {e}")))?;
    let mut temp = NamedTempFile::new()
        .map_err(|e| DcfError::Other(format!("failed to create temp image: {e}")))?;
    {
        let rgba = image.to_rgba8();
        PngEncoder::new(temp.as_file_mut())
            .write_image(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
                ColorType::Rgba8.into(),
            )
            .map_err(|e| DcfError::Other(format!("failed to encode image for ocr: {e}")))?;
        temp.flush()
            .map_err(|e| DcfError::Other(format!("failed to flush temp image: {e}")))?;
    }
    let temp_path = temp.into_temp_path();
    let path_buf = temp_path.to_path_buf();
    let path_str = path_buf
        .to_str()
        .ok_or_else(|| DcfError::Other("temp image path not valid UTF-8".to_string()))?
        .to_string();
    if !tess.set_image(&path_str) {
        let _ = temp_path.close();
        return Err(DcfError::Other(
            "failed to load image into tesseract".to_string(),
        ));
    }
    let text = tess
        .get_utf8_text()
        .map_err(|e| DcfError::Other(format!("tesseract failed: {e}")))?;
    let _ = temp_path.close();
    Ok(text)
}

#[cfg(not(feature = "ocr"))]
#[allow(dead_code)]
pub fn image_to_text(_image: &DynamicImage, _languages: &[String]) -> Result<String> {
    Err(DcfError::OcrSupportDisabled)
}
