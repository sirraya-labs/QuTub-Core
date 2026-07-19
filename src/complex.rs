//! Complex number arithmetic used throughout the simulator.

use std::fmt;
use std::ops::Neg;

/// Numerical tolerance used for equality/threshold checks across the crate.
pub(crate) const EPSILON: f64 = 1e-12;

/// Largest register size this simulator will construct (2^16 amplitudes).
pub(crate) const MAX_QUBITS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self::new(self.real * factor, self.imag * factor)
    }

    pub fn exp(&self) -> Self {
        let exp_real = self.real.exp();
        Self::new(
            exp_real * self.imag.cos(),
            exp_real * self.imag.sin()
        )
    }

    pub fn is_nan(&self) -> bool {
        self.real.is_nan() || self.imag.is_nan()
    }

    pub fn is_finite(&self) -> bool {
        self.real.is_finite() && self.imag.is_finite()
    }

    pub fn real(&self) -> f64 {
        self.real
    }

    pub fn imag(&self) -> f64 {
        self.imag
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.real, -self.imag)
    }
}

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

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imag - other.imag)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        self.scale(scalar)
    }
}

impl std::ops::Div<f64> for Complex {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        self.scale(1.0 / scalar)
    }
}
