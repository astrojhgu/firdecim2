use num::traits::AsPrimitive;
use num::{Complex, Zero};

/// Decimate-by-2 halfband FIR filter
/// Optimized for branch-free SIMD via continuous window.
#[inline]
pub fn resample2_complex<S, T>(
    input: &[Complex<S>],
    output: &mut [Complex<T>],
    coeff: &[T],
    state: &mut [Complex<T>],
    bit_shift: u32,
) -> usize
where
    S: Copy + AsPrimitive<T>,
    T: 'static
        + Copy
        + Zero
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Shr<u32, Output = T>,
{
    let ntaps = coeff.len();
    assert!(ntaps % 2 == 1);
    let mid = ntaps / 2;
    let state_len = ntaps - 1;
    assert_eq!(state.len(), state_len);
    assert!(input.len() >= state_len);

    assert!(input.len() % 2 == 0);
    let n_out = input.len() / 2;
    assert!(output.len() >= n_out);

    // 1) Pre-convert input S -> T once.
    // 使用 collect 代替循环，语义更清晰。
    let input_t: Vec<Complex<T>> = input
        .iter()
        .map(|x| Complex::new(x.re.as_(), x.im.as_()))
        .collect();

    // 2) Re-introduce branch-free window: [state, input_t]
    // 这是实现 SIMD 友好的、无分支访问的关键。分配和复制是值得的。
    let mut window: Vec<Complex<T>> = Vec::with_capacity(state_len + input_t.len());
    window.extend_from_slice(&state[..]); // copy state into window
    window.extend_from_slice(&input_t[..]); // then the converted input
    let window_slice = window.as_slice(); // 使用切片避免重复边界检查

    // 3) Pre-extract odd taps (offsets and coeffs). (保持不变)
    let mut odd_offsets: Vec<usize> = Vec::new();
    let mut odd_coeffs: Vec<T> = Vec::new();
    for k in (1..=mid).step_by(2) {
        odd_offsets.push(k);
        odd_coeffs.push(coeff[mid - k]);
    }
    let n_odd = odd_offsets.len();

    // 4) Core loop: for each output i
    // 目标: 增强循环展开，提高数据局部性，让编译器自动向量化。
    for i in 0..n_out {
        // pos 索引现在直接作用于连续的 window_slice，没有分支。
        let pos = 2 * i + mid;

        // 中心抽头
        let center = window_slice[pos];
        let mut acc_re = coeff[mid] * center.re;
        let mut acc_im = coeff[mid] * center.im;

        // 迭代奇数抽头，使用您的 4x 展开
        let mut j = 0usize;
        // 使用 unsafe + get_unchecked 来消除内层循环的边界检查，
        // 从而最大化自动向量化的机会。这是性能提升的关键一步。
        unsafe {
            let odd_offsets_ptr = odd_offsets.as_ptr();
            let odd_coeffs_ptr = odd_coeffs.as_ptr();
            let window_ptr = window_slice.as_ptr();

            while j + 3 < n_odd {
                // 批量加载系数和偏移量
                let k0 = *odd_offsets_ptr.add(j);
                let k1 = *odd_offsets_ptr.add(j + 1);
                let k2 = *odd_offsets_ptr.add(j + 2);
                let k3 = *odd_offsets_ptr.add(j + 3);

                let c0 = *odd_coeffs_ptr.add(j);
                let c1 = *odd_coeffs_ptr.add(j + 1);
                let c2 = *odd_coeffs_ptr.add(j + 2);
                let c3 = *odd_coeffs_ptr.add(j + 3);

                // 批量加载数据 (pos +/- k 是安全的，前提是调用者传入了足够的输入)
                let x0l = *window_ptr.add(pos - k0);
                let x0r = *window_ptr.add(pos + k0);
                let x1l = *window_ptr.add(pos - k1);
                let x1r = *window_ptr.add(pos + k1);
                let x2l = *window_ptr.add(pos - k2);
                let x2r = *window_ptr.add(pos + k2);
                let x3l = *window_ptr.add(pos - k3);
                let x3r = *window_ptr.add(pos + k3);

                // 乘加操作
                acc_re += c0 * (x0l.re + x0r.re);
                acc_im += c0 * (x0l.im + x0r.im);

                acc_re += c1 * (x1l.re + x1r.re);
                acc_im += c1 * (x1l.im + x1r.im);

                acc_re += c2 * (x2l.re + x2r.re);
                acc_im += c2 * (x2l.im + x2r.im);

                acc_re += c3 * (x3l.re + x3r.re);
                acc_im += c3 * (x3l.im + x3r.im);

                j += 4;
            }

            // 尾部处理 (不变)
            while j < n_odd {
                let k = *odd_offsets_ptr.add(j);
                let c = *odd_coeffs_ptr.add(j);
                let xl = *window_ptr.add(pos - k);
                let xr = *window_ptr.add(pos + k);

                acc_re += c * (xl.re + xr.re);
                acc_im += c * (xl.im + xr.im);
                j += 1;
            }
        } // end unsafe block

        output[i] = Complex::new(acc_re >> bit_shift, acc_im >> bit_shift);
    }

    // 5) Update state: copy last `state_len` samples from input_t
    // (保持不变，已是高效的切片复制)
    let input_start_idx = input_t.len() - state_len;
    state.copy_from_slice(&input_t[input_start_idx..]);

    n_out
}

#[cfg(test)]
mod tests {
    use super::resample2_complex;
    use num::Complex;
    use num::traits::FloatConst;
    use num::traits::Zero;

    fn fir_coeffs() -> Vec<i32> {
        vec![
            -26, 0, 28, 0, -31, 0, 36, 0, -44, 0, 53, 0, -65, 0, 79, 0, -97, 0, 117, 0, -140, 0,
            167, 0, -198, 0, 234, 0, -274, 0, 319, 0, -371, 0, 429, 0, -496, 0, 572, 0, -661, 0,
            764, 0, -887, 0, 1036, 0, -1219, 0, 1454, 0, -1768, 0, 2212, 0, -2897, 0, 4112, 0,
            -6917, 0, 20848, 32767, 20848, 0, -6917, 0, 4112, 0, -2897, 0, 2212, 0, -1768, 0, 1454,
            0, -1219, 0, 1036, 0, -887, 0, 764, 0, -661, 0, 572, 0, -496, 0, 429, 0, -371, 0, 319,
            0, -274, 0, 234, 0, -198, 0, 167, 0, -140, 0, 117, 0, -97, 0, 79, 0, -65, 0, 53, 0,
            -44, 0, 36, 0, -31, 0, 28, 0, -26,
        ]
    }

    #[test]
    fn unit_pulse() {
        let fir_coeffs = fir_coeffs();

        let mut state: Vec<_> = vec![Complex::zero(); fir_coeffs.len() - 1];
        let mut input = vec![Complex::<i8>::zero(); fir_coeffs.len() + 1];
        input[0] = Complex::new(1, 0);
        let mut output = vec![Complex::zero(); input.len() / 2];
        resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
        fir_coeffs
            .as_slice()
            .iter()
            .step_by(2)
            .zip(output.iter())
            .for_each(|(&a, &b)| {
                assert_eq!(a, b.re);
                println!("{} {}", a, b.re);
            });
    }

    #[test]
    fn boundary_test() {
        let fir_coeffs = fir_coeffs();

        let mut state: Vec<Complex<i32>> = vec![Complex::zero(); fir_coeffs.len() - 1];
        let input: Vec<_> = (0..8192)
            .map(|i| {
                Complex::new(
                    (127.0 * ((i as f64 / 128.0) * 2.0 * f64::PI()).sin()) as i8,
                    0,
                )
            })
            .collect();
        let mut output1 = vec![Complex::zero(); input.len() / 2];

        resample2_complex(&input, &mut output1, &fir_coeffs, &mut state, 0);
        let mut state: Vec<Complex<i32>> = vec![Complex::zero(); fir_coeffs.len() - 1];

        let mut output2 = vec![Complex::zero(); input.len() / 2];
        resample2_complex(
            &input[0..4000],
            &mut output2[0..2000],
            &fir_coeffs,
            &mut state,
            0,
        );

        resample2_complex(
            &input[4000..],
            &mut output2[2000..],
            &fir_coeffs,
            &mut state,
            0,
        );
        output1.iter().zip(output2.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
            println!("{} {}", a, b);
        });
    }

    #[test]
    fn test_zero_input() {
        let fir_coeffs = fir_coeffs();
        let ntaps = fir_coeffs.len();
        let state_len = ntaps - 1;

        let mut state = vec![Complex::new(0i32, 0); state_len];
        let input_len = 2 * state_len; // 确保输入较长
        let input = vec![Complex::<i8>::zero(); input_len]; // 输入全为零

        let n_out = input_len / 2;
        let mut output = vec![Complex::new(-99i32, -99); n_out]; // 用非零值初始化输出

        resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);

        // 期望：输出所有样本都为零
        for (i, sample) in output.iter().enumerate() {
            assert!(
                sample.re.is_zero() && sample.im.is_zero(),
                "Output sample {} should be zero, but was: {}",
                i,
                sample
            );
        }

        // 期望：状态在处理完零输入后也应该全为零
        for (i, sample) in state.iter().enumerate() {
            assert!(
                sample.re.is_zero() && sample.im.is_zero(),
                "State sample {} should be zero, but was: {}",
                i,
                sample
            );
        }
    }

    #[test]
    fn test_dc_response() {
        let fir_coeffs = fir_coeffs();
        let ntaps = fir_coeffs.len();
        let state_len = ntaps - 1;

        // S=i32, T=i32
        let input_val = Complex::new(32_i32, 0); // 输入1.0对应的定点数 (Q16.16)
        let bit_shift = 16; // 目标缩放，将 65535 增益映射到约 1.0

        let mut state = vec![Complex::zero(); state_len];
        let input_len = 1024; // 长输入序列以保证稳态
        let input = vec![input_val; input_len]; // 输入恒定值 65536
        let n_out = input_len / 2;
        let mut output = vec![Complex::zero(); n_out];

        resample2_complex(&input, &mut output, &fir_coeffs, &mut state, bit_shift);

        // 预期的稳定输出值：
        // output = input_val * (Total Gain) / 2^bit_shift
        // output = 65536 * 65535 / 65536 = 65535
        let expected_re = 32;
        let tolerance = 1; // 允许微小的定点计算误差

        // 检查最后一部分样本是否达到稳态
        let start_check = n_out / 2;
        for (i, sample) in output.iter().skip(start_check).enumerate() {
            assert!(
                (sample.re - expected_re).abs() <= tolerance,
                "Output sample {} (re) was {}, expected {} ±{}",
                i,
                sample.re,
                expected_re,
                tolerance
            );
            assert!(
                sample.im.is_zero(),
                "Output sample {} (im) was not zero: {}",
                i,
                sample.im
            );
        }
    }

    #[test]
    fn test_minimal_input_fixed() {
        let fir_coeffs = fir_coeffs();
        let ntaps = fir_coeffs.len();
        let state_len = ntaps - 1;

        type T_ACC = i64;
        let fir_coeffs_t: Vec<T_ACC> = fir_coeffs.iter().map(|&c| c as T_ACC).collect();

        // *** 关键修正: 确保输入长度 L_in >= state_len ***
        // 我们选择 input.len() = state_len，即只产生 (state_len / 2) 个输出
        let input_len = state_len;
        let n_out = state_len / 2;

        // 1. 设置一个非零初始状态 (延迟线中的数据)
        let initial_state_val = Complex::new(100i64, -50);
        let mut state = vec![initial_state_val; state_len];

        // 2. 设置输入: 长度为 state_len，所有样本为 input_val
        let input_val = Complex::new(20i32, 20); // S=i32
        let input = vec![input_val; input_len];
        let mut output = vec![Complex::zero(); n_out]; // Output (T_ACC)

        let bit_shift = 0; // 不缩放

        resample2_complex(&input, &mut output, &fir_coeffs_t, &mut state, bit_shift);

        // 3. 验证输出:
        // 预期输出是 state 尾部 + input 头部混合后的结果，难以精确计算，
        // 但我们可以验证它既不是零也不是固定的 DC 响应。
        assert!(n_out > 0);
        assert!(
            output[0].re != 0,
            "Output should be non-zero after mixed input/state."
        );

        // 4. 验证状态更新:
        // 由于 input.len() = state_len, input_start_idx = state_len - state_len = 0.
        // 新状态应该完全由 input_t[0..state_len] 复制而来，即全部是 input_val 的转换。
        let expected_state_val = Complex::new(input_val.re as i64, input_val.im as i64);
        for (i, sample) in state.iter().enumerate() {
            assert_eq!(
                *sample, expected_state_val,
                "State sample {} after update should be the converted input value.",
                i
            );
        }
    }
}
