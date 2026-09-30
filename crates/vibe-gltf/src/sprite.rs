//! Sprite sheets: frame rectangles, timing and UVs.
//!
//! A sheet is described separately from its image, so a sheet can be built and
//! checked without decoding a PNG. The arithmetic a sheet needs — grid slicing,
//! frame timing, UV rectangles — is where the mistakes live, so it is all pure
//! functions over numbers.

use serde::{Deserialize, Serialize};

use crate::error::GltfError;

/// How a sheet's frames are arranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SheetLayout {
    /// Cut into a grid of equal cells, row by row.
    #[default]
    Grid,
    /// Every frame is the same size, in one row.
    Horizontal,
    /// Every frame is the same size, in one column.
    Vertical,
}

/// How frames are laid out on the source image.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SheetDesc {
    /// Columns for a grid; frames for a strip.
    pub columns: u32,
    /// Rows for a grid; frames for a vertical strip.
    pub rows: u32,
    /// Which arrangement to use.
    pub layout: SheetLayout,
    /// Seconds each frame is shown, if the caller has not set a duration.
    pub frame_seconds: f32,
    /// Whether playback restarts after the last frame.
    pub looping: bool,
}

impl SheetDesc {
    /// A grid of `columns` by `rows`, at 10 frames per second.
    pub fn grid(columns: u32, rows: u32) -> SheetDesc {
        SheetDesc {
            columns,
            rows,
            layout: SheetLayout::Grid,
            frame_seconds: 0.1,
            looping: true,
        }
    }

    /// A horizontal strip of `count` frames.
    pub fn strip(count: u32) -> SheetDesc {
        SheetDesc {
            columns: count,
            rows: 1,
            layout: SheetLayout::Horizontal,
            frame_seconds: 0.1,
            looping: true,
        }
    }

    /// A grid at a given frame rate.
    pub fn with_fps(mut self, fps: f32) -> SheetDesc {
        self.frame_seconds = if fps > 0.0 { 1.0 / fps } else { 0.1 };
        self
    }

    /// A grid that plays once.
    pub fn once(mut self) -> SheetDesc {
        self.looping = false;
        self
    }

    /// How many frames the arrangement yields.
    pub fn frame_count(&self) -> usize {
        match self.layout {
            SheetLayout::Grid => (self.columns as usize) * (self.rows as usize),
            SheetLayout::Horizontal | SheetLayout::Vertical => self.columns as usize,
        }
    }
}

/// One frame's rectangle in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FrameRect {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl FrameRect {
    /// The right edge, exclusive.
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// The bottom edge, exclusive.
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// True when this rectangle lies inside an image of the given size.
    pub fn fits(&self, image_width: u32, image_height: u32) -> bool {
        self.right() <= image_width && self.bottom() <= image_height && self.width > 0
    }

    /// The four UV coordinates, in the order bottom-left, bottom-right,
    /// top-right, top-left.
    ///
    /// V is flipped because glTF and Vulkan put the image origin at the top
    /// left while a quad's UV origin is the bottom left. Getting this backwards
    /// renders the sheet upside down, which is easy to miss when the frames
    /// happen to be symmetric.
    pub fn uvs(&self, image_width: u32, image_height: u32) -> [[f32; 2]; 4] {
        if image_width == 0 || image_height == 0 {
            return [[0.0; 2]; 4];
        }
        let (u0, v0) = (self.x as f32, self.y as f32);
        let (u1, v1) = (self.right() as f32, self.bottom() as f32);
        let u0 = u0 / image_width as f32;
        let u1 = u1 / image_width as f32;
        // Flip V: pixel row 0 is the top of the image, and V 0 is the bottom.
        let top = 1.0 - v0 / image_height as f32;
        let bottom = 1.0 - v1 / image_height as f32;
        [[u0, bottom], [u1, bottom], [u1, top], [u0, top]]
    }
}

/// Cut a grid into frame rectangles.
///
/// The cell size is the floor of the division, and the remainder is dropped
/// rather than spread across the cells: a fractional cell produces a frame that
/// is one pixel narrower than its neighbour, and the seam shows as a flickering
/// line when the sheet is animated.
pub fn grid_frames(image_width: u32, image_height: u32, columns: u32, rows: u32) -> Vec<FrameRect> {
    if columns == 0 || rows == 0 {
        return Vec::new();
    }
    let cell_w = image_width / columns;
    let cell_h = image_height / rows;
    if cell_w == 0 || cell_h == 0 {
        return Vec::new();
    }
    let mut frames = Vec::with_capacity((columns * rows) as usize);
    for row in 0..rows {
        for col in 0..columns {
            frames.push(FrameRect {
                x: col * cell_w,
                y: row * cell_h,
                width: cell_w,
                height: cell_h,
            });
        }
    }
    frames
}

/// Cut a horizontal strip into equal frames.
pub fn strip_frames_horizontal(image_width: u32, image_height: u32, count: u32) -> Vec<FrameRect> {
    if count == 0 {
        return Vec::new();
    }
    let cell_w = image_width / count;
    if cell_w == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| FrameRect {
            x: i * cell_w,
            y: 0,
            width: cell_w,
            height: image_height,
        })
        .collect()
}

/// Cut a vertical strip into equal frames.
pub fn strip_frames_vertical(image_width: u32, image_height: u32, count: u32) -> Vec<FrameRect> {
    if count == 0 {
        return Vec::new();
    }
    let cell_h = image_height / count;
    if cell_h == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| FrameRect {
            x: 0,
            y: i * cell_h,
            width: image_width,
            height: cell_h,
        })
        .collect()
}

/// Cut a sheet's frames according to its layout.
pub fn slice(desc: &SheetDesc, image_width: u32, image_height: u32) -> Vec<FrameRect> {
    match desc.layout {
        SheetLayout::Grid => grid_frames(image_width, image_height, desc.columns, desc.rows),
        SheetLayout::Horizontal => strip_frames_horizontal(image_width, image_height, desc.columns),
        SheetLayout::Vertical => strip_frames_vertical(image_width, image_height, desc.columns),
    }
}

/// Check that a sheet's frames fit its image.
///
/// Every frame has to be checked, not just the first: a grid whose image does
/// not divide evenly produces a last cell that runs past the edge, and that is
/// the frame that samples whatever is next in memory.
pub fn validate_sheet(
    name: &str,
    desc: &SheetDesc,
    image_width: u32,
    image_height: u32,
) -> Result<Vec<FrameRect>, GltfError> {
    if desc.frame_count() == 0 {
        return Err(GltfError::NoFrames {
            name: name.to_string(),
            frames: desc.frame_count(),
        });
    }
    let frames = slice(desc, image_width, image_height);
    if frames.is_empty() {
        return Err(GltfError::NoFrames {
            name: name.to_string(),
            frames: desc.frame_count(),
        });
    }
    for (i, f) in frames.iter().enumerate() {
        if !f.fits(image_width, image_height) {
            return Err(GltfError::FrameOutOfBounds {
                name: name.to_string(),
                frame: i,
                rect: format!("{}x{}+{}+{}", f.width, f.height, f.x, f.y),
                width: image_width,
                height: image_height,
            });
        }
    }
    Ok(frames)
}

/// Which frame a sheet's clock is showing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SheetClock {
    /// The current time in seconds.
    pub time: f32,
    /// Seconds per frame.
    pub frame_seconds: f32,
    /// How many frames the sheet has.
    pub frame_count: usize,
    /// Whether to loop at the end.
    pub looping: bool,
}

impl SheetClock {
    /// A clock for a sheet.
    pub fn new(desc: &SheetDesc, frame_count: usize) -> SheetClock {
        SheetClock {
            time: 0.0,
            frame_seconds: desc.frame_seconds,
            frame_count,
            looping: desc.looping,
        }
    }

    /// The frame index the clock is on.
    ///
    /// A zero frame time or a zero frame count would divide by zero, so both
    /// return frame 0 rather than a NaN index.
    pub fn frame(&self) -> usize {
        if self.frame_count == 0 || self.frame_seconds <= 0.0 {
            return 0;
        }
        let raw = (self.time / self.frame_seconds).floor();
        if !raw.is_finite() {
            return 0;
        }
        let index = raw as i64;
        if self.looping {
            index.rem_euclid(self.frame_count as i64) as usize
        } else {
            index.clamp(0, self.frame_count as i64 - 1) as usize
        }
    }

    /// Advance the clock.
    pub fn advance(&mut self, delta: f32) {
        self.time += delta;
    }

    /// The frame's UV rectangle on an image of the given size.
    pub fn uv(&self, frames: &[FrameRect], image_width: u32, image_height: u32) -> [[f32; 2]; 4] {
        frames
            .get(self.frame())
            .map(|f| f.uvs(image_width, image_height))
            .unwrap_or([[0.0; 2]; 4])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_yields_one_frame_per_cell() {
        let f = grid_frames(64, 32, 4, 2);
        assert_eq!(f.len(), 8);
        assert_eq!(
            f[0],
            FrameRect {
                x: 0,
                y: 0,
                width: 16,
                height: 16
            }
        );
        assert_eq!(
            f[1],
            FrameRect {
                x: 16,
                y: 0,
                width: 16,
                height: 16
            }
        );
    }

    #[test]
    fn a_grid_fills_row_by_row() {
        let f = grid_frames(64, 64, 4, 4);
        // The second row starts at y = 16, not x = 48.
        assert_eq!(f[4].y, 16);
        assert_eq!(f[4].x, 0);
    }

    #[test]
    fn a_grid_that_does_not_divide_evenly_drops_the_remainder() {
        // 100/3 is 33.33; spreading the remainder gives frames of differing
        // width and a visible seam.
        let f = grid_frames(100, 100, 3, 3);
        assert!(f.iter().all(|r| r.width == 33), "{f:?}");
        assert_eq!(f[2].right(), 99, "the last pixel column is unused");
    }

    #[test]
    fn a_grid_smaller_than_its_cell_count_yields_nothing() {
        assert!(grid_frames(2, 2, 4, 4).is_empty());
    }

    #[test]
    fn a_grid_with_a_zero_dimension_yields_nothing() {
        assert!(grid_frames(64, 64, 0, 4).is_empty());
        assert!(grid_frames(64, 64, 4, 0).is_empty());
    }

    #[test]
    fn a_horizontal_strip_cuts_evenly() {
        let f = strip_frames_horizontal(80, 16, 4);
        assert_eq!(f.len(), 4);
        assert!(f.iter().all(|r| r.width == 20 && r.height == 16));
    }

    #[test]
    fn a_vertical_strip_cuts_evenly() {
        let f = strip_frames_vertical(16, 80, 4);
        assert!(f.iter().all(|r| r.width == 16 && r.height == 20));
    }

    #[test]
    fn a_strip_with_no_frames_yields_nothing() {
        assert!(strip_frames_horizontal(80, 16, 0).is_empty());
        assert!(strip_frames_vertical(16, 80, 0).is_empty());
    }

    #[test]
    fn a_strip_narrower_than_its_frame_count_yields_nothing() {
        assert!(strip_frames_horizontal(3, 16, 4).is_empty());
    }

    #[test]
    fn slicing_follows_the_layout() {
        let image = (64, 64);
        let grid = SheetDesc::grid(4, 4);
        assert_eq!(slice(&grid, image.0, image.1).len(), 16);

        let strip = SheetDesc::strip(4);
        assert_eq!(slice(&strip, image.0, image.1).len(), 4);
    }

    #[test]
    fn a_grid_desc_counts_its_cells() {
        assert_eq!(SheetDesc::grid(4, 3).frame_count(), 12);
    }

    #[test]
    fn a_strip_desc_counts_its_frames() {
        assert_eq!(SheetDesc::strip(7).frame_count(), 7);
    }

    #[test]
    fn frames_per_second_becomes_a_frame_time() {
        let d = SheetDesc::grid(2, 2).with_fps(20.0);
        assert!((d.frame_seconds - 0.05).abs() < 1e-6);
    }

    #[test]
    fn a_zero_frame_rate_keeps_a_usable_default() {
        let d = SheetDesc::grid(2, 2).with_fps(0.0);
        assert!(d.frame_seconds > 0.0);
    }

    #[test]
    fn a_sheet_can_be_marked_non_looping() {
        assert!(!SheetDesc::grid(2, 2).once().looping);
    }

    #[test]
    fn a_valid_sheet_passes() {
        let d = SheetDesc::grid(4, 4);
        assert!(validate_sheet("hero", &d, 64, 64).is_ok());
    }

    #[test]
    fn a_sheet_with_no_frames_is_rejected() {
        let d = SheetDesc {
            columns: 0,
            rows: 0,
            ..SheetDesc::grid(4, 4)
        };
        let err = validate_sheet("empty", &d, 64, 64).unwrap_err();
        assert!(matches!(err, GltfError::NoFrames { frames: 0, .. }));
    }

    #[test]
    fn a_sheet_too_small_for_its_grid_is_rejected() {
        // 4x4 cells in a 2x2 image: each cell would be zero-sized.
        let d = SheetDesc::grid(4, 4);
        let err = validate_sheet("tiny", &d, 2, 2).unwrap_err();
        assert!(matches!(err, GltfError::NoFrames { .. }), "{err:?}");
    }

    #[test]
    fn an_oversized_frame_is_reported_with_its_number() {
        // 5 columns into 10 pixels gives cells of width 2, which fits; force
        // the out-of-bounds path with a hand-made rectangle.
        let f = FrameRect {
            x: 8,
            y: 0,
            width: 4,
            height: 4,
        };
        assert!(!f.fits(10, 10));
    }

    #[test]
    fn a_frame_at_the_exact_edge_fits() {
        let f = FrameRect {
            x: 8,
            y: 6,
            width: 2,
            height: 4,
        };
        assert!(f.fits(10, 10), "the right edge is exclusive");
    }

    #[test]
    fn a_zero_width_frame_does_not_fit() {
        let f = FrameRect {
            x: 0,
            y: 0,
            width: 0,
            height: 4,
        };
        assert!(!f.fits(10, 10));
    }

    #[test]
    fn frame_uvs_span_the_rectangle() {
        // Half the image's width, so U runs from 0 to 0.5.
        let f = FrameRect {
            x: 0,
            y: 0,
            width: 32,
            height: 64,
        };
        let uvs = f.uvs(64, 64);
        assert!((uvs[0][0]).abs() < 1e-6, "{uvs:?}");
        assert!((uvs[1][0] - 0.5).abs() < 1e-6, "{uvs:?}");
    }

    #[test]
    fn frame_uvs_flip_v_for_the_image_origin() {
        // The top of the rectangle maps to V = 1 and its bottom to V = 0.5,
        // because a quad's UV origin is the bottom left while the image's is
        // the top left. Getting this backwards renders the sheet upside down.
        let f = FrameRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let uvs = f.uvs(64, 64);
        let top_row = uvs[2][1];
        let bottom_row = uvs[0][1];
        assert!((top_row - 1.0).abs() < 1e-6, "top {top_row}");
        assert!((bottom_row - 0.5).abs() < 1e-6, "bottom {bottom_row}");
    }

    #[test]
    fn frame_uvs_on_a_zero_sized_image_are_zero() {
        let f = FrameRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        assert_eq!(f.uvs(0, 0), [[0.0; 2]; 4]);
    }

    #[test]
    fn frame_uvs_are_in_corner_order() {
        // bottom-left, bottom-right, top-right, top-left
        let f = FrameRect {
            x: 16,
            y: 16,
            width: 32,
            height: 32,
        };
        let uvs = f.uvs(64, 64);
        assert_eq!(uvs[0], [0.25, 0.25]);
        assert_eq!(uvs[1], [0.75, 0.25]);
        assert_eq!(uvs[2], [0.75, 0.75]);
        assert_eq!(uvs[3], [0.25, 0.75]);
    }

    #[test]
    fn a_clock_starts_on_the_first_frame() {
        let c = SheetClock::new(&SheetDesc::grid(4, 1), 4);
        assert_eq!(c.frame(), 0);
    }

    #[test]
    fn a_clock_advances_a_frame_at_a_time() {
        let d = SheetDesc::grid(4, 1).with_fps(10.0);
        let mut c = SheetClock::new(&d, 4);
        c.advance(0.05);
        assert_eq!(c.frame(), 0, "half a frame in is still the first frame");
        c.advance(0.06);
        assert_eq!(c.frame(), 1);
    }

    #[test]
    fn a_looping_clock_wraps() {
        let d = SheetDesc::grid(4, 1).with_fps(10.0);
        let mut c = SheetClock::new(&d, 4);
        // 0.45s at 10fps is frame 4, which is past the last of four frames, so
        // a looping sheet shows frame 0.
        c.advance(0.45);
        assert_eq!(c.frame(), 0, "past the end it wraps");
        // 0.51s at 10fps is frame 5, which wraps to 1.
        c.advance(0.06);
        assert_eq!(c.frame(), 1);
    }

    #[test]
    fn a_non_looping_clock_holds_the_last_frame() {
        let d = SheetDesc::grid(4, 1).with_fps(10.0).once();
        let mut c = SheetClock::new(&d, 4);
        c.advance(99.0);
        assert_eq!(c.frame(), 3);
    }

    #[test]
    fn a_looping_clock_cycles_every_frame_in_order() {
        let d = SheetDesc::grid(4, 1).with_fps(10.0);
        let mut c = SheetClock::new(&d, 4);
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(c.frame());
            c.advance(0.1);
        }
        assert_eq!(seen, vec![0, 1, 2, 3], "one frame per tick, in order");
    }

    #[test]
    fn a_clock_with_no_frames_is_on_frame_zero() {
        let c = SheetClock::new(&SheetDesc::grid(4, 1), 0);
        assert_eq!(c.frame(), 0);
    }

    #[test]
    fn a_clock_with_a_zero_frame_time_does_not_divide_by_zero() {
        let d = SheetDesc {
            frame_seconds: 0.0,
            ..SheetDesc::grid(4, 1)
        };
        let mut c = SheetClock::new(&d, 4);
        c.advance(1.0);
        assert_eq!(c.frame(), 0);
    }

    #[test]
    fn a_clock_with_a_non_finite_time_is_on_frame_zero() {
        let mut c = SheetClock::new(&SheetDesc::grid(4, 1), 4);
        c.time = f32::INFINITY;
        assert_eq!(c.frame(), 0);
    }

    #[test]
    fn a_clock_reports_the_frames_uvs() {
        let d = SheetDesc::grid(2, 1).with_fps(10.0);
        let frames = slice(&d, 64, 64);
        let mut c = SheetClock::new(&d, frames.len());
        assert_eq!(c.uv(&frames, 64, 64), frames[0].uvs(64, 64));
        c.advance(0.11);
        assert_eq!(c.uv(&frames, 64, 64), frames[1].uvs(64, 64));
    }

    #[test]
    fn a_clock_with_no_frame_list_reports_zero_uvs() {
        let c = SheetClock::new(&SheetDesc::grid(2, 1), 0);
        assert_eq!(c.uv(&[], 64, 64), [[0.0; 2]; 4]);
    }
}
