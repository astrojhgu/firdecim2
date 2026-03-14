#![feature(portable_simd)]

use firdecim2::I32s;
use firdecim2::{fir::fir_coeffs, firdec_worker::resample2};
const N_BATCH: usize = 512;

pub fn main() {
    let fir_coeffs = fir_coeffs();
    // 调用优化后的函数
    let fir_coeffs_i32: Vec<std::simd::Simd<i32, 16>> =
        fir_coeffs.iter().map(|&c| I32s::splat(c as i32)).collect();

    let n_tap_half = fir_coeffs.len();
    let m_half = n_tap_half - 1;

    // input 长度 512 个 i16，代表 256 个 Complex
    let mut input = vec![0i16; N_BATCH];

    // 设置第一个复数为 1 + 1i
    input[4] = 1; // I0
    input[5] = 1; // Q0

    // 状态空间：(ntaps - 1) * 2 是历史复数点占用的 i16 数量
    let n_old_state = m_half * 2 * 2;
    let mut state = vec![0i16; n_old_state + N_BATCH];

    // 输出长度减半
    let mut output = vec![0i16; N_BATCH / 2];

    resample2(&input, &mut output, &fir_coeffs, &mut state, 0);

    for i in (0..output.len()).step_by(2) {
        println!("{} {}", output[i], output[i + 1]);
    }
}
