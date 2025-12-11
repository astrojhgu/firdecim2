use rayon::prelude::*;
use std::simd::{Simd, num::SimdInt};

/// =============================================
/// SIMD LANE 数，统一在此修改
/// =============================================
pub const LANES: usize = 16;

/// 2:1 half-band decimator, using lane-parameterized SIMD and Rayon  
/// 要求：input.len() == 2 * output.len()，且 output.len() % LANES == 0
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
    assert_eq!(n_output % LANES, 0);
    assert_eq!(n_input % (LANES * 2), 0);

    // 状态更新
    state[n_old_state..required_state_len].copy_from_slice(input);

    let state_slice = state.as_ref();
    let coeff_slice = coeff.as_ref();

    // 统一 lane 类型
    type I16s = Simd<i16, LANES>;
    type I32s = Simd<i32, LANES>;

    let center_coeff = I32s::splat(coeff[0] as i32);
    let shift_vec = I32s::splat(bit_shift as i32);

    // 每 chunk = LANES 输出样本
    output.par_chunks_mut(LANES).enumerate().for_each(|(chunk_idx, out_chunk)| {
        let out_idx = chunk_idx * LANES;
        let mut acc = I32s::splat(0);

        // ---------------------------
        // 中心 tap h[0]
        // ---------------------------
        let mut center_buf = [0i32; LANES];
        for i in 0..LANES {
            center_buf[i] = state_slice[(2 * (out_idx + i)) + m_half] as i32;
        }
        acc += I32s::from_array(center_buf) * center_coeff;

        // ---------------------------
        // 奇数 tap (1,3,5,...)
        // ---------------------------
        for k in (1..=m_half).step_by(2) {
            let c = I32s::splat(coeff_slice[k] as i32);

            let mut pos_buf = [0i32; LANES];
            let mut neg_buf = [0i32; LANES];

            for i in 0..LANES {
                let center = (2 * (out_idx + i)) + m_half;
                pos_buf[i] = state_slice[center + k] as i32;
                neg_buf[i] = state_slice[center - k] as i32;
            }

            let sum = I32s::from_array(pos_buf) + I32s::from_array(neg_buf);
            acc += sum * c;
        }

        // ---------------------------
        // 右移（不饱和）
        // ---------------------------
        let shifted = acc >> shift_vec;

        // ---------------------------
        // 写回输出
        // ---------------------------
        for i in 0..LANES {
            out_chunk[i] = shifted[i] as i16;
        }
    });

    // ---------------------------
    // 状态更新
    // ---------------------------
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
