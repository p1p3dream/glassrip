//! JPEG decoding to 8-bit gray (luma) and interleaved BGR.
//!
//! Baseline files go through [`crate::jpeg`], which reproduces libjpeg-turbo's default output
//! bit for bit (the decoder behind OpenCV's `imread`). Other files (progressive, arithmetic
//! coded, unusual sampling) fall back to `zune-jpeg`, whose IDCT, chroma upsampling and color
//! conversion differ from libjpeg-turbo by a few levels on some pixels.

use std::path::Path;

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::error::{MediaError, Result};
use crate::jpeg::JpegError;
use crate::plane::{Bgr, Plane};

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| MediaError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn decode_with(path: &Path, bytes: &[u8], cs: ColorSpace) -> Result<(usize, usize, Vec<u8>)> {
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(cs)
        .set_max_width(1 << 15)
        .set_max_height(1 << 15);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    let err = |e: &dyn std::fmt::Display| MediaError::Decode {
        path: path.to_path_buf(),
        message: e.to_string(),
    };
    let pixels = dec.decode().map_err(|e| err(&e))?;
    let (w, h) = dec
        .dimensions()
        .ok_or_else(|| err(&"missing dimensions after decode"))?;
    let cn = match cs {
        ColorSpace::Luma => 1,
        _ => 3,
    };
    if pixels.len() != w * h * cn {
        return Err(err(&"decoded buffer has unexpected length"));
    }
    Ok((w, h, pixels))
}

/// Decodes a JPEG file to its 8-bit luma plane.
///
/// This corresponds to `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`, which asks libjpeg for
/// `JCS_GRAYSCALE` output and therefore returns the Y component without color conversion.
pub fn decode_gray(path: &Path) -> Result<Plane<u8>> {
    let bytes = read(path)?;
    decode_gray_bytes(path, &bytes)
}

/// Decodes in-memory JPEG bytes to luma. `path` is used only for error messages.
pub fn decode_gray_bytes(path: &Path, bytes: &[u8]) -> Result<Plane<u8>> {
    let (w, h, px) = match crate::jpeg::decode(bytes) {
        Ok(d) => (d.width, d.height, d.gray()),
        Err(JpegError::Unsupported(_)) => decode_with(path, bytes, ColorSpace::Luma)?,
        Err(JpegError::Corrupt(m)) => {
            return Err(MediaError::Decode {
                path: path.to_path_buf(),
                message: m,
            })
        }
    };
    Plane::from_vec(w, h, px).ok_or(MediaError::Size {
        width: w,
        height: h,
        reason: "luma buffer length mismatch",
    })
}

/// Decodes a JPEG file to interleaved BGR, corresponding to `cv2.imread(path)`.
pub fn decode_bgr(path: &Path) -> Result<Bgr> {
    let bytes = read(path)?;
    decode_bgr_bytes(path, &bytes)
}

/// Decodes in-memory JPEG bytes to BGR. `path` is used only for error messages.
pub fn decode_bgr_bytes(path: &Path, bytes: &[u8]) -> Result<Bgr> {
    let exact = crate::jpeg::decode(bytes).and_then(|d| Ok((d.width, d.height, d.bgr()?)));
    let (width, height, data) = match exact {
        Ok(v) => v,
        Err(JpegError::Unsupported(_)) => decode_with(path, bytes, ColorSpace::BGR)?,
        Err(JpegError::Corrupt(m)) => {
            return Err(MediaError::Decode {
                path: path.to_path_buf(),
                message: m,
            })
        }
    };
    Ok(Bgr {
        width,
        height,
        data,
    })
}
