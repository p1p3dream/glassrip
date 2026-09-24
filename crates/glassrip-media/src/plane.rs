//! Minimal row-major image containers.

/// A single-channel image stored row-major without padding.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane<T> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major pixel data, `width * height` entries.
    pub data: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    /// Creates a plane filled with `T::default()`.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![T::default(); width * height],
        }
    }

    /// Creates a plane filled with `value`.
    pub fn filled(width: usize, height: usize, value: T) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Wraps existing data. Returns `None` when the length does not match.
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Option<Self> {
        (data.len() == width * height).then_some(Self {
            width,
            height,
            data,
        })
    }

    /// Pixel at column `x`, row `y`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.width + x]
    }

    /// Mutable pixel at column `x`, row `y`.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[y * self.width + x]
    }

    /// Row `y` as a slice.
    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    /// Applies `f` to every pixel, producing a new plane.
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Plane<U> {
        Plane {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }
}

/// An interleaved 3-channel 8-bit image in B, G, R order (OpenCV's default layout).
#[derive(Debug, Clone, PartialEq)]
pub struct Bgr {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Interleaved BGR data, `3 * width * height` bytes.
    pub data: Vec<u8>,
}

impl Bgr {
    /// Pixel `(b, g, r)` at column `x`, row `y`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [u8; 3] {
        let i = 3 * (y * self.width + x);
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}
