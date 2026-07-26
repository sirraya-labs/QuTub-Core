//! Complex number arithmetic used throughout the simulator.
//!
//! A minimal, dependency-free complex number type (rather than pulling in
//! `num-complex`) sized for exactly what this crate needs: amplitude and
//! matrix-entry arithmetic for state vectors and density matrices up to
//! [`MAX_QUBITS`] qubits.
//!
//! # Example
//!
//! ```
//! use sirraya_qutub::Complex;
//!
//! let a = Complex::new(1.0, 2.0);
//! let b = Complex::i();
//! let product = a * b; // (1+2i) * i = -2+i
//! assert!((product.real() - (-2.0)).abs() < 1e-12);
//! assert!((product.imag() - 1.0).abs() < 1e-12);
//! ```

use std::fmt;
use std::ops::Neg;

/// Numerical tolerance used for equality/threshold checks across the crate.
pub(crate) const EPSILON: f64 = 1e-12;

/// Largest register size this simulator will construct (2^16 amplitudes).
pub(crate) const MAX_QUBITS: usize = 16;

/// A complex number `real + imag*i`, stored as two `f64` fields.
///
/// Implements the arithmetic operators (`+`, `-`, `*`, unary `-`, and
/// `*`/`/` by a real `f64` scalar) needed for quantum amplitude and
/// matrix-entry arithmetic elsewhere in this crate. `Copy` since it's two
/// `f64`s -- cheaper to pass by value than to borrow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    /// Constructs `real + imag*i`.
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    /// The additive identity, `0 + 0i`.
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// The multiplicative identity, `1 + 0i`.
    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    /// The imaginary unit, `0 + 1i`.
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Squared magnitude `|z|^2 = real^2 + imag^2`. Prefer this over
    /// `magnitude()` when only comparing magnitudes (e.g. measurement
    /// probabilities), since it skips the square root.
    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    /// Magnitude (modulus) `|z| = sqrt(real^2 + imag^2)`.
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Complex conjugate `real - imag*i`.
    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }

    /// Scales both components by a real `factor`: `factor * (real + imag*i)`.
    /// Equivalent to, and used to implement, the `Mul<f64>` operator.
    pub fn scale(&self, factor: f64) -> Self {
        Self::new(self.real * factor, self.imag * factor)
    }

    /// Complex exponential `e^z = e^real * (cos(imag) + i*sin(imag))`
    /// (Euler's formula). Used to build phase and rotation gates -- e.g.
    /// `Complex::new(0.0, angle).exp()` is the unit-magnitude phase factor
    /// `e^{i*angle}`.
    pub fn exp(&self) -> Self {
        let exp_real = self.real.exp();
        Self::new(
            exp_real * self.imag.cos(),
            exp_real * self.imag.sin()
        )
    }

    /// `true` if either component is NaN.
    pub fn is_nan(&self) -> bool {
        self.real.is_nan() || self.imag.is_nan()
    }

    /// `true` if both components are finite (not NaN or +/-infinity).
    pub fn is_finite(&self) -> bool {
        self.real.is_finite() && self.imag.is_finite()
    }

    /// The real component.
    pub fn real(&self) -> f64 {
        self.real
    }

    /// The imaginary component.
    pub fn imag(&self) -> f64 {
        self.imag
    }
}

/// Unary negation: `-z = -real - imag*i`.
impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.real, -self.imag)
    }
}

/// Formats as `a+bi` / `a-bi`, dropping the real or imaginary part
/// when it is negligible (below `EPSILON`) and omitting the `+` sign
/// before a negative imaginary part. Fixed at 6 decimal places.
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.imag.abs() < EPSILON {
            write!(f, "{:.6}", self.real)
        } else if self.real.abs() < EPSILON {
            write!(f, "{:.6}i", self.imag)
        } else if self.imag > 0.0 {
            write!(f, "{:.6}+{:.6}i", self.real, self.imag)
        } else {
            write!(f, "{:.6}{:.6}i", self.real, self.imag)
        }
    }
}

/// Complex addition, component-wise.
impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }
}

/// Complex subtraction, component-wise.
impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imag - other.imag)
    }
}

/// Complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    }
}

/// Scales by a real `f64`; see [`Complex::scale`].
impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        self.scale(scalar)
    }
}

/// Divides by a real `f64` (multiplies by `1.0 / scalar`); see
/// [`Complex::scale`].
impl std::ops::Div<f64> for Complex {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        self.scale(1.0 / scalar)
    }
}