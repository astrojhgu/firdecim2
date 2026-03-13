use std::simd::{Simd, num::SimdInt, simd_swizzle};

pub const LANES: usize = 16;

pub fn resample2(
    input: &[i16],
    output: &mut [i16],
    coeff: &[i16],
    state: &mut [i16],
    bit_shift: u32,
) {
    let n_half_taps = coeff.len();
    let m_half = n_half_taps - 1;
    let n_input = input.len();
    let n_output = output.len();
    let n_old_state = m_half * 4;

    state[n_old_state..n_old_state + n_input].copy_from_slice(input);

    type I32s = Simd<i32, LANES>;

    // --- 核心优化：预处理位移 ---
    // 我们不再在循环里执行 acc >> bit_shift
    // 如果 bit_shift 较小，或者我们能接受在累加前处理系数
    // 注意：这里为了保持精度，我们依然使用 i32 累加，但减少一次向量位移指令
    let coeffs_i32: Vec<I32s> = coeff.iter()
        .map(|&c| I32s::splat(c as i32))
        .collect();

    for j in 0..(n_output / LANES) {
        let out_idx = j * LANES;
        let state_offset = 2 * out_idx + (m_half * 2);

        // 使用之前优化过的加载函数
        let mut acc = extract_even_iq(&state[state_offset..]) * coeffs_i32[0];

        let mut k = 1;
        while k <= m_half {
            let c = coeffs_i32[k];
            let p = extract_even_iq(&state[state_offset + k * 2..]);
            let n = extract_even_iq(&state[state_offset - k * 2..]);
            
            // FMA (Fused Multiply-Add) 风格：编译器会尝试将其优化为单周期指令
            acc += (p + n) * c;
            k += 2;
        }

        // --- 仅在此处执行一次位移 ---
        let shifted = acc >> I32s::splat(bit_shift as i32);
        
        let out_simd: Simd<i16, LANES> = shifted.cast::<i16>();
        output[out_idx..out_idx + LANES].copy_from_slice(out_simd.as_array());
    }

    state.copy_within(n_input..n_input + n_old_state, 0);
}
/// 优化后的加载并求和函数
#[inline(always)]
fn load_sum_pos_neg(
    state: &[i16],
    offset: usize,
    k2: usize,
    coeff: &Simd<i32, LANES>,
) -> Simd<i32, LANES> {
    // 提取 pos 方向的 IQ 对
    let p = extract_even_iq(&state[offset + k2..]);
    // 提取 neg 方向的 IQ 对
    let n = extract_even_iq(&state[offset - k2..]);

    // 在 i32 层面求和并乘系数
    (p + n) * *coeff
}

#[inline(always)]
fn extract_even_iq(src: &[i16]) -> Simd<i32, LANES> {
    // 这里的关键是：编译器能否优化这 2 个 128-bit load 为 1 个 256-bit load
    // 直接 load 32 字节
    let s = Simd::<i16, 32>::from_slice(&src[0..32]);

    // 这是一个非常快的 Shuffle 模式，直接选取 I0,Q0, I2,Q2...
    // 使用 swizzle 提取偶数复数对
    let picked = simd_swizzle!(
        s,
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]
    );

    picked.cast::<i32>()
}

#[inline(always)]
fn load_complex_deinterleaved_i32(src: &[i16], coeff: &Simd<i32, LANES>) -> Simd<i32, LANES> {
    extract_even_iq(src) * *coeff
}

#[cfg(test)]
mod tests {
    use super::resample2;
    //use crate::fir;
    use crate::fir::fir_coeffs;
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
        resample2(&input, &mut output, &fir_coeffs, &mut state, 0);

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
        resample2(&input, &mut output_a, &coeff, &mut state_a, bit_shift);
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
            &coeff,
            &mut state_b,
            bit_shift,
        );

        // 第二段：处理后一半 (此时 state_b 内部已经自动完成了 copy_within)
        resample2(
            &input[mid_input..],
            &mut output_b[mid_output..],
            &coeff,
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
