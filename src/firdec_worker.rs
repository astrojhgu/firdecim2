use std::simd::{Simd, num::SimdInt, simd_swizzle};

use crate::{LANES, I32s};


#[inline(always)]
pub fn resample2(
    input: &[i16],
    output: &mut [i16],
    coeffs_i32: &[I32s],
    state: &mut [i16],
    bit_shift: u32,
) {
    let n_half_taps = coeffs_i32.len();
    let m_half = n_half_taps - 1;
    let n_input = input.len();
    let n_output = output.len();
    let n_old_state = m_half * 4;

    state[n_old_state..n_old_state + n_input].copy_from_slice(input);

    let shift_vec = I32s::splat(bit_shift as i32);

    for j in 0..(n_output / LANES) {
        let out_idx = j * LANES;
        let state_offset = 2 * out_idx + (m_half * 2);

        // --- 中心 Tap ---
        let mut acc0 = extract_even_iq(&state[state_offset..]) * coeffs_i32[0];
        let mut acc1 = I32s::splat(0); // 第二个累加器，打破流水线依赖

        // --- 展开循环 (每步处理 2 个 tap，即 4 个对称点) ---
        let mut k = 1;
        while k + 2 <= m_half {
            // 第一组
            let c_a = coeffs_i32[k];
            let p_a = extract_even_iq(&state[state_offset + k * 2..]);
            let n_a = extract_even_iq(&state[state_offset - k * 2..]);
            acc0 += (p_a + n_a) * c_a;

            // 第二组
            let c_b = coeffs_i32[k + 2];
            let p_b = extract_even_iq(&state[state_offset + (k + 2) * 2..]);
            let n_b = extract_even_iq(&state[state_offset - (k + 2) * 2..]);
            acc1 += (p_b + n_b) * c_b;

            k += 4; // 步进 4
        }

        // 处理剩余的 k (如果有)
        while k <= m_half {
            let c = coeffs_i32[k];
            let p = extract_even_iq(&state[state_offset + k * 2..]);
            let n = extract_even_iq(&state[state_offset - k * 2..]);
            acc0 += (p + n) * c;
            k += 2;
        }

        // 合并累加器
        let acc = acc0 + acc1;

        let shifted = acc >> shift_vec;
        let out_simd: Simd<i16, LANES> = shifted.cast::<i16>();
        output[out_idx..out_idx + LANES].copy_from_slice(out_simd.as_array());
    }

    state.copy_within(n_input..n_input + n_old_state, 0);
}

// 保持这个高效的 swizzle 不变，但确保它内联
#[inline(always)]
fn extract_even_iq(src: &[i16]) -> Simd<i32, LANES> {
    // 强制使用对齐加载（如果可能）或者直接 from_slice
    let s = Simd::<i16, 32>::from_slice(&src[0..32]);
    let picked = simd_swizzle!(
        s,
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]
    );
    picked.cast::<i32>()
}

#[cfg(test)]
mod tests {
    use super::resample2;
    //use crate::fir;
    use crate::{fir::fir_coeffs, I32s, LANES};
    //use num::Complex;
    //use num::traits::FloatConst;
    //use num::traits::Zero;
    //use std::fs::File;
    //use std::io::Write;
    const N_BATCH: usize = 512;

    #[test]
    fn unit_pulse_complex() {
        let fir_coeffs = fir_coeffs(); // 假设这是半带滤波器的前一半系数（含中心点）

        // 生成完整的滤波器系数用于比对
        // 注意：半带滤波器的偶数项（除了中心点）通常为 0
        let fir_coeffs_full: Vec<i16> = fir_coeffs
            .iter()
            .rev()
            .chain(fir_coeffs.iter().skip(1))
            .cloned()
            .collect();

        let n_tap_half = fir_coeffs.len();
        let m_half = n_tap_half - 1;

        // input 长度 512 个 i16，代表 256 个 Complex
        let mut input = vec![0i16; N_BATCH];

        // 设置第一个复数为 1 + 1i
        input[0] = 1; // I0
        input[1] = 1; // Q0

        // 状态空间：(ntaps - 1) * 2 是历史复数点占用的 i16 数量
        let n_old_state = m_half * 2 * 2;
        let mut state = vec![0i16; n_old_state + N_BATCH];

        // 输出长度减半
        let mut output = vec![0i16; N_BATCH / 2];

        // 调用优化后的函数
        let fir_coeffs_i32:Vec<std::simd::Simd<i32, 16>>=fir_coeffs.iter()
        .map(|&c| I32s::splat(c as i32))
        .collect();

        resample2(&input, &mut output, &fir_coeffs_i32, &mut state, 0);

        // --- 验证逻辑 ---
        // 对于单位脉冲 [1, 1, 0, 0, ...]，输出应该是滤波器的系数
        // 但因为是 2:1 降采样，输出只会保留偶数项的响应
        // 预期输出序列应该是：[h[0], h[0], h[2], h[2], h[4], h[4] ...] (如果脉冲在位置0)
        // 注意：h[k] 对应 fir_coeffs_full 中的值

        fir_coeffs_full
            .iter()
            .step_by(2) // 降采样 2 对应的系数步进
            .enumerate()
            .for_each(|(idx, &expected_val)| {
                let out_re = output[idx * 2]; // 输出的 I
                let out_im = output[idx * 2 + 1]; // 输出的 Q

                println!(
                    "TapIdx {}: Expected {}, Got I={}, Q={}",
                    idx, expected_val, out_re, out_im
                );

                assert_eq!(out_re, expected_val, "实部不匹配 @ index {}", idx);
                assert_eq!(out_im, expected_val, "虚部不匹配 @ index {}", idx);
            });
    }

    #[test]
    fn test_segmented_consistency() {
        use crate::fir::fir_coeffs; // 假设你的系数生成函数在此

        // 1. 准备参数
        let coeff = fir_coeffs();
        let fir_coeffs_i32:Vec<std::simd::Simd<i32, 16>>=coeff.iter()
        .map(|&c| I32s::splat(c as i32))
        .collect();

        let n_half_taps = coeff.len();
        let m_half = n_half_taps - 1;
        let n_old_state = 2 * m_half;
        let bit_shift = 2; // 示例位移

        // 构造一段足够长的随机输入数据 (必须是 LANES*2 的倍数)
        let total_input_len = 512;
        let input: Vec<i16> = (0..total_input_len as i16).collect();

        // --- 实验组 A: 一次性处理 ---
        let mut state_a = vec![0i16; n_old_state * 2 + total_input_len];
        let mut output_a = vec![0i16; total_input_len / 2];
        resample2(&input, &mut output_a, &fir_coeffs_i32, &mut state_a, bit_shift);
        println!("output a: {:?}", output_a);

        // --- 实验组 B: 分两段处理 ---
        let mut state_b = vec![0i16; n_old_state * 2 + total_input_len / 2]; // 状态空间需足够容纳单次输入
        let mut output_b = vec![0i16; total_input_len / 2];

        let mid_input = total_input_len / 2; // 从中间切分
        let mid_output = mid_input / 2;

        // 第一段：处理前一半
        // 注意：state 的长度在 resample2 中有断言检查，传入的 state 切片长度必须符合约定
        resample2(
            &input[..mid_input],
            &mut output_b[..mid_output],
            &fir_coeffs_i32,
            &mut state_b,
            bit_shift,
        );

        // 第二段：处理后一半 (此时 state_b 内部已经自动完成了 copy_within)
        resample2(
            &input[mid_input..],
            &mut output_b[mid_output..],
            &fir_coeffs_i32,
            &mut state_b,
            bit_shift,
        );

        println!("output b: {:?}", output_b);

        // --- 验证结果 ---
        // 检查 output_a 和 output_b 是否逐元素相等
        assert_eq!(output_a.len(), output_b.len(), "输出长度不一致");
        for i in 0..output_a.len() {
            assert_eq!(
                output_a[i], output_b[i],
                "分段处理在索引 {} 处不一致！A: {}, B: {}",
                i, output_a[i], output_b[i]
            );
        }
        println!("分段等效性测试通过！");
    }
}
