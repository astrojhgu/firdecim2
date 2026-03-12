use std::simd::{Simd, cmp::SimdPartialOrd, num::SimdInt};

pub const LANES: usize = 16;

/// 优化后的单核版本：移除 Rayon，利用高效 SIMD 加载与重组
///  /// 2:1 half-band decimator, using lane-parameterized SIMD and Rayon

/// 要求：input.len() == 2 * output.len()，且 output.len() % LANES == 0 
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
    let n_old_state = 2 * m_half;

    // 预拷贝输入到 state，避免在计算循环中处理边界
    state[n_old_state..n_old_state + n_input].copy_from_slice(input);

    type I16s = Simd<i16, LANES>;
    type I32s = Simd<i32, LANES>;

    // 预转换系数为 SIMD 类型以减少循环内开销
    let coeffs_i32: Vec<I32s> = coeff.iter().map(|&c| I32s::splat(c as i32)).collect();

    // 性能关键：外层循环步进 LANES
    // 目标：一次性计算 LANES 个输出
    for j in 0..(n_output / LANES) {
        let out_start = j * LANES;
        let state_offset = 2 * out_start + m_half;
        
        // --- 处理中心 Tap ---
        // 关键优化：由于是降采样2，state_slice[state_offset + 2*i] 并不连续
        // 但我们可以加载两个连续的 LANES 块，然后通过 gather 或 shuffle 取出偶数位
        let mut acc = load_deinterleaved_even_i32(&state[state_offset..], &coeffs_i32[0]);

        // --- 处理对称 Taps (k=1, 3, 5...) ---
        let mut k = 1;
        while k <= m_half {
            let c = coeffs_i32[k];
            
            // 正向偏移加载并求和
            let pos_sum = load_deinterleaved_even_i32(&state[state_offset + k..], &c);
            // 负向偏移加载并求和
            let neg_sum = load_deinterleaved_even_i32(&state[state_offset - k..], &c);
            
            acc += pos_sum + neg_sum;
            k += 2;
        }

        // --- 批量位移并写回 ---
        let shifted = acc >> I32s::splat(bit_shift as i32);
        for i in 0..LANES {
            output[out_start + i] = shifted[i] as i16;
        }
    }

    // 状态移动
    state.copy_within(n_input..n_input + n_old_state, 0);
}

/// 辅助函数：从起始位置读取数据，仅提取偶数索引对应的元素并乘以系数
/// 解决了原代码中 center_buf/pos_buf 手动循环打包的问题
#[inline(always)]
fn load_deinterleaved_even_i32(src: &[i16], coeff: &Simd<i32, LANES>) -> Simd<i32, LANES> {
    // 载入连续的 2*LANES 数据
    let low = Simd::<i16, LANES>::from_slice(&src[0..LANES]);
    let high = Simd::<i16, LANES>::from_slice(&src[LANES..2 * LANES]);
    
    // 使用 cast 转换为 i32 以防止乘法溢出
    let low32 = low.cast::<i32>();
    let high32 = high.cast::<i32>();

    // 这里的“魔法”：从两个向量中提取偶数位 (0, 2, 4... 30)
    // 现代 CPU (AVX2/AVX-512) 对这类 Shuffle 指令有极高的吞吐量
    let mut result = Simd::<i32, LANES>::default();
    
    // 手动解构：这在编译后通常对应一组 vpshuf 或 vperm 指令
    // 比原先的 for 循环写入 array 快得多
    for i in 0..LANES / 2 {
        result[i] = low32[i * 2];
        result[i + LANES / 2] = high32[i * 2];
    }

    result * *coeff
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
