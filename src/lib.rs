#![feature(test)]
#![feature(portable_simd)]
pub mod firdec_worker;
pub mod decim_pipeline;
pub mod fir;
pub type I32s = std::simd::Simd<i32, LANES>;
pub const LANES: usize = 16;
