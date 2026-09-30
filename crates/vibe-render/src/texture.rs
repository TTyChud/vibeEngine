//! Loading an image off disk and getting it onto the GPU.
//!
//! Decoding is pure and lives here; the upload is a set of Vulkan calls over
//! `vibe_vk`'s buffer and image types, because that is where the barrier
//! knowledge already is. The split matters: "this JPEG is 8-bit and needs a
//! format conversion" is a question a test can answer with no device, while
//! "the image is in the layout the descriptor names" is not.
//!
//! Only PNG and JPEG are decoded. Both are what a user actually has on disk, and
//! supporting one more format is a feature flag rather than new code, since the
//! path below hands the decoded RGBA to the same upload either way.

use std::path::Path;

use ash::vk;

use crate::error::RenderError;

/// A decoded image, ready to upload.
///
/// RGBA8 with rows top to bottom, which is the layout the box geometry's UVs
/// assume: `v = 0` is the top of the image. This is the opposite of Vulkan's
/// texture coordinate convention, so the upload does *not* flip — a flip here
/// would need a matching flip in the geometry, and the geometry's UVs are the
/// one thing that is easier to read and reason about when they match what the
/// image looks like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 texels, row by row from the top.
    pub pixels: Vec<u8>,
}

impl DecodedImage {
    /// Bytes one row occupies.
    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    /// Bytes a full upload needs.
    pub fn byte_len(&self) -> usize {
        self.stride() * self.height as usize
    }

    /// A 1x1 opaque white image, which is what an untextured box binds.
    ///
    /// The mesh shader multiplies its lighting by the texture unconditionally, so
    /// an untextured box needs *some* texture: white makes the multiply a no-op,
    /// which is the whole reason this exists rather than branching in the shader.
    pub fn white() -> DecodedImage {
        DecodedImage {
            width: 1,
            height: 1,
            pixels: vec![255, 255, 255, 255],
        }
    }
}

/// Why an image could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// The file could not be read.
    Io(String),
    /// The bytes are not an image format this build decodes.
    Decode(String),
    /// The image has a zero dimension, which Vulkan rejects.
    Empty,
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::Io(e) => write!(f, "could not read the image: {e}"),
            ImageError::Decode(e) => write!(f, "could not decode the image: {e}"),
            ImageError::Empty => write!(f, "the image has no pixels"),
        }
    }
}

impl std::error::Error for ImageError {}

/// True when this build can decode the extension.
///
/// A `.webp` or `.tga` next to the PNGs shows up in a content browser and is
/// then un-clickable, which reads as a broken editor rather than an unsupported
/// format, so the browser needs to know the difference.
pub fn is_supported(path: &Path) -> bool {
    match extension(path).as_deref() {
        Some("png") | Some("jpg") | Some("jpeg") => true,
        _ => false,
    }
}

/// The lowercase extension of a path, without the dot.
pub fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

/// Decode an image file to RGBA8.
///
/// Every input format becomes RGBA8 rather than keeping its own layout, because
/// the upload path and the descriptor expect one format. An RGB JPEG costs three
/// extra bytes per texel to convert, which is the price of not having a second
/// texture path to keep correct.
pub fn load(path: &Path) -> Result<DecodedImage, ImageError> {
    let bytes = std::fs::read(path).map_err(|e| ImageError::Io(e.to_string()))?;
    decode(&bytes)
}

/// Decode image bytes to RGBA8.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    let image = image::load_from_memory(bytes).map_err(|e| ImageError::Decode(e.to_string()))?;
    // `to_rgba8` is the conversion: whatever the source was, the result is four
    // bytes per texel, which is what `vk::Format::R8G8B8A8_UNORM` samples. The
    // dimensions come off the converted buffer rather than the source image, so
    // there is no second shape to keep in step.
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Empty);
    }
    Ok(DecodedImage {
        width,
        height,
        pixels: rgba.into_raw(),
    })
}

/// The Vulkan format an image is uploaded as.
///
/// One format for everything, so the shader and the descriptor do not branch on
/// how the image happened to be encoded. `R8G8B8A8_UNORM` rather than
/// `R8G8B8A8_SRGB`: the format is chosen by what the decoder produced, and
/// writing sRGB encoding into a UNORM image would double-encode every texture
/// loaded from a JPEG.
pub const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The format an image is uploaded as.
pub const fn texture_format() -> vk::Format {
    TEXTURE_FORMAT
}

impl From<ImageError> for RenderError {
    fn from(e: ImageError) -> RenderError {
        RenderError::Pipeline(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2 PNG, built by hand so the test has no fixture file to keep.
    ///
    /// Encoded as a minimal PNG: signature, an IHDR, one IDAT of zlib-compressed
    /// scanlines, and an IEND. `image`'s encoder is behind its `png` feature
    /// only for decoding in this build, so the bytes are written out directly.
    fn two_by_two_png() -> Vec<u8> {
        // Each scanline is a filter byte followed by two RGBA texels, so a
        // 2-wide image is 1 + 8 bytes per row and 18 bytes in total.
        let raw = [
            0u8, 255, 0, 0, 255, 0, 255, 0, 255, // red, green
            0, 0, 0, 255, 255, 255, 255, 0, 255, // blue, transparent white
        ];
        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes()); // width
        ihdr.extend_from_slice(&2u32.to_be_bytes()); // height
        ihdr.push(8); // bit depth
        ihdr.push(6); // colour type: RGBA
        ihdr.extend_from_slice(&[0, 0, 0]); // compression, filter, interlace
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &zlib_store(&raw));
        push_chunk(&mut png, b"IEND", &[]);
        png
    }

    fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        // A real CRC, not a filler byte: the decoder checks it, so a wrong one
        // would fail the load for a reason that has nothing to do with the code
        // under test.
        let mut crc_input = Vec::with_capacity(4 + data.len());
        crc_input.extend_from_slice(kind);
        crc_input.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    }

    /// A zlib stream of stored (uncompressed) deflate blocks.
    ///
    /// Stored blocks need no compressor, which is the point: this is a test
    /// fixture, and a hand-rolled inflate would be a second thing to get wrong.
    fn zlib_store(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01]; // zlib header: deflate, no dictionary
        let mut offset = 0;
        while offset < data.len() {
            let len = (data.len() - offset).min(0xffff);
            let last = if offset + len >= data.len() { 1 } else { 0 };
            out.push(last);
            out.extend_from_slice(&(len as u16).to_le_bytes());
            out.extend_from_slice(&(!(len as u16)).to_le_bytes()); // one's complement
            out.extend_from_slice(&data[offset..offset + len]);
            offset += len;
        }
        // Adler-32 of the uncompressed data, which zlib requires.
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in data {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        out.extend_from_slice(&((b << 16) | a).to_be_bytes());
        out
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    #[test]
    fn a_png_decodes_to_its_pixels() {
        let image = decode(&two_by_two_png()).expect("the fixture is a valid png");
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.pixels.len(), 2 * 2 * 4, "RGBA8, four bytes a texel");
    }

    #[test]
    fn the_decoded_pixels_are_in_top_to_bottom_order() {
        // The fixture's first scanline is red and its second is green, so the
        // first texel is red only if rows run the way the UVs assume. A decoder
        // that flipped the image would put green first, which shows up as every
        // texture on a box being upside down.
        let image = decode(&two_by_two_png()).expect("valid png");
        assert_eq!(
            &image.pixels[0..4],
            &[255, 0, 0, 255],
            "the top row's first texel should be red"
        );
        assert_eq!(
            &image.pixels[4..8],
            &[0, 255, 0, 255],
            "the top row's second texel should be green"
        );
        assert_eq!(
            &image.pixels[8..12],
            &[0, 0, 255, 255],
            "the second row's first texel should be blue"
        );
    }

    #[test]
    fn an_rgba8_png_keeps_its_alpha() {
        // A transparent texel must stay transparent, or a cut-out texture loses
        // its holes and the box comes out as a solid rectangle.
        let mut png = Vec::new();
        png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.push(8);
        ihdr.push(6);
        ihdr.extend_from_slice(&[0, 0, 0]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &zlib_store(&[0, 10, 20, 30, 0]));
        push_chunk(&mut png, b"IEND", &[]);
        let image = decode(&png).expect("valid png");
        assert_eq!(image.pixels, vec![10, 20, 30, 0], "alpha was lost");
    }

    #[test]
    fn garbage_bytes_are_a_decode_error_not_a_panic() {
        // A user pointing the content browser at a text file must get a message,
        // not a crash: this is the path an unknown file type takes.
        let err = decode(b"this is not an image").expect_err("garbage must not decode");
        assert!(matches!(err, ImageError::Decode(_)), "got {err:?}");
    }

    #[test]
    fn an_empty_buffer_is_a_decode_error() {
        assert!(matches!(decode(&[]), Err(ImageError::Decode(_))));
    }

    #[test]
    fn a_missing_file_is_an_io_error() {
        let err = load(Path::new("/definitely/not/here.png")).expect_err("no such file");
        assert!(matches!(err, ImageError::Io(_)), "got {err:?}");
    }

    #[test]
    fn the_stride_and_length_follow_the_dimensions() {
        let image = decode(&two_by_two_png()).expect("valid png");
        assert_eq!(image.stride(), 8);
        assert_eq!(image.byte_len(), 16);
        assert_eq!(image.byte_len(), image.pixels.len());
    }

    #[test]
    fn the_white_placeholder_is_one_opaque_texel() {
        let w = DecodedImage::white();
        assert_eq!((w.width, w.height), (1, 1));
        assert_eq!(w.pixels, vec![255, 255, 255, 255]);
        // Opaque, because the mesh shader writes texel.a to the framebuffer and
        // a placeholder with alpha 0 would make an untextured box invisible.
        assert_eq!(w.pixels[3], 255);
    }

    #[test]
    fn png_and_jpeg_are_supported() {
        for name in ["a.png", "a.PNG", "a.jpg", "a.jpeg", "a.JPEG"] {
            assert!(is_supported(Path::new(name)), "{name} should load");
        }
    }

    #[test]
    fn other_formats_are_not_claimed_as_supported() {
        for name in ["a.webp", "a.tga", "a.exr", "a.txt", "noextension"] {
            assert!(!is_supported(Path::new(name)), "{name} is not decoded");
        }
    }

    #[test]
    fn the_extension_is_lowercased_without_the_dot() {
        assert_eq!(extension(Path::new("a.PNG")).as_deref(), Some("png"));
        assert_eq!(extension(Path::new("dir/b.JpG")).as_deref(), Some("jpg"));
        assert_eq!(extension(Path::new("a")), None);
    }

    #[test]
    fn the_upload_format_is_rgba8_unorm() {
        // The shader and the decoder both assume this; a change here has to
        // change the conversion in `decode` too.
        assert_eq!(texture_format(), vk::Format::R8G8B8A8_UNORM);
    }
}
