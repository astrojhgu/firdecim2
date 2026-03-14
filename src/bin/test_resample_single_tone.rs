#![feature(portable_simd)]

use firdecim2::I32s;
use firdecim2::{fir::fir_coeffs, firdec_worker::resample2_plain as resample2};
const N_BATCH: usize = 2048;

pub fn main() {
    let fir_coeffs = fir_coeffs();
    // 调用优化后的函数
    let _fir_coeffs_i32: Vec<std::simd::Simd<i32, 16>> =
        fir_coeffs.iter().map(|&c| I32s::splat(c as i32)).collect();

    let n_tap_half = fir_coeffs.len();
    let m_half = n_tap_half - 1;

    let omega = 0.1;

    // 状态空间：(ntaps - 1) * 2 是历史复数点占用的 i16 数量
    let n_old_state = m_half * 2 * 2;
    let mut state = vec![0i16; n_old_state + N_BATCH];

    // 输出长度减半
    // input 长度 512 个 i16，代表 256 个 Complex
    //let mut input = vec![0i16; N_BATCH];
    let input: Vec<_> = (0..N_BATCH)
        .map(|i| {
            let phase = (i / 2) as f64 * omega;
            if i % 2 == 0 {
                (1024.0 * (phase.cos())) as i16
            } else {
                (1024.0 * (phase.sin())) as i16
            }
        })
        .collect();

    let mut output = vec![0i16; N_BATCH / 2];

    resample2(&input, &mut output, &fir_coeffs, &mut state, 16);

    for i in 0..N_BATCH / 4{
        println!("{} {}", output[i*2], output[i*2+1]);
    }
}
