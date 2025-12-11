use rayon::prelude::*;
use std::simd::{Simd, num::SimdInt};

/// 16-lane SIMD + Rayon 并行迭代 halfband decimator
pub fn resample2(
    input: &[i16],
    output: &mut [i16],
    coeff: &[i16],
    state: &mut [i16],
    bit_shift: u32,
) {
    let n_half_taps = coeff.len();
    assert!(n_half_taps >= 1);
    let m_half = n_half_taps - 1;

    let n_input = input.len();
    let n_output = output.len();
    let n_old_state = 2 * m_half;
    let required_state_len = n_old_state + n_input;

    assert_eq!(n_input, n_output * 2);
    assert_eq!(state.len(), required_state_len);
    assert_eq!(n_output % 16, 0);
    assert_eq!(n_input % 32, 0);

    // 状态拷贝
    state[n_old_state..required_state_len].copy_from_slice(input);
    let state_slice = state.as_ref();
    let coeff_slice = coeff.as_ref();

    type I16x16 = Simd<i16, 16>;
    type I64x16 = Simd<i64, 16>;

    let center_coeff = I64x16::splat(coeff[0] as i64);
    let shift_vec = I64x16::splat(bit_shift as i64);

    // 并行迭代 output 块，每个块 16 个输出
    output.par_chunks_mut(16).enumerate().for_each(|(chunk_idx, out_chunk)| {
        let out_idx = chunk_idx * 16;
        let mut acc = I64x16::splat(0);

        // --- 中心 tap h[0] ---
        let mut center_vals = [0i64; 16];
        for i in 0..16 {
            center_vals[i] = state_slice[(2 * (out_idx + i)) + m_half] as i64;
        }
        acc += I64x16::from_array(center_vals) * center_coeff;

        // --- 奇数 taps ---
        for k in (1..=m_half).step_by(2) {
            let c = I64x16::splat(coeff_slice[k] as i64);

            let mut pos_vals = [0i64; 16];
            let mut neg_vals = [0i64; 16];

            for i in 0..16 {
                let center_idx = (2 * (out_idx + i)) + m_half;
                pos_vals[i] = state_slice[center_idx + k] as i64;
                neg_vals[i] = state_slice[center_idx - k] as i64;
            }

            let sum = I64x16::from_array(pos_vals) + I64x16::from_array(neg_vals);
            acc += sum * c;
        }

        // 右移缩放（不饱和）
        let shifted = acc >> shift_vec;

        // 写回 output
        for i in 0..16 {
            out_chunk[i] = shifted[i] as i16;
        }
    });

    // 更新 state
    state.copy_within(n_input..required_state_len, 0);
}


#[cfg(test)]
mod tests {
    use super::resample2;
    use crate::fir;
    use crate::fir::fir_coeffs;
    use num::Complex;
    use num::traits::FloatConst;
    use num::traits::Zero;
    use std::fs::File;
    use std::io::Write;
    const N_BATCH: usize = 512;

    #[test]
    fn unit_pulse() {
        let fir_coeffs = fir_coeffs();
        let fir_coeffs_full = fir_coeffs
            .iter()
            .rev()
            .chain(fir_coeffs.iter().skip(1))
            .cloned()
            .collect::<Vec<_>>();
        println!("{:?}", fir_coeffs_full);
        let n_tap_half = fir_coeffs.len();
        let n_tap_full = n_tap_half * 2 - 1;

        let mut state: Vec<_> = vec![0; N_BATCH + n_tap_full - 1];
        let mut input = vec![0; N_BATCH];
        input[0] = 1;
        let mut output = vec![0; input.len() / 2];
        println!("{:?}", fir_coeffs_full);
        resample2(&input, &mut output, &fir_coeffs, &mut state, 0);
        fir_coeffs_full
            .iter()
            .step_by(2)
            .zip(output.iter())
            .for_each(|(a, b)| {
                println!("{} {}", a, b);
                assert_eq!(a, b);
            });
    }
}
