use num::{Complex, Zero};
use num::traits::AsPrimitive;

/// Decimate-by-2 halfband FIR filter
/// input:  &[Complex<S>]
/// output: &mut [Complex<T>]
/// coeff:  &[T]  (halfband FIR, mid tap == 0.5 or similar)
/// state:  filter delay line, len = coeff.len() - 1

/// Halfband FIR decimate-by-2 for complex IQ samples
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
    let mid = ntaps / 2;
    let state_len = ntaps - 1;

    assert!(ntaps % 2 == 1);
    assert!(input.len() % 2 == 0);
    assert_eq!(state.len(), state_len);

    let n_out = input.len() / 2;

    for i in 0..n_out {
        let pos = 2 * i + mid;       // 相对于 “state + input” 的逻辑 pos
        let in_base = pos as isize - state_len as isize;

        // 中心 sample
        let xc = if in_base < 0 {
            state[pos]
        } else {
            let x = &input[in_base as usize];
            Complex::new(x.re.as_(), x.im.as_())
        };

        let mut acc_re = coeff[mid] * xc.re;
        let mut acc_im = coeff[mid] * xc.im;

        // 奇数 taps
        for k in (1..=mid).step_by(2) {
            let idx_l = pos - k;
            let idx_r = pos + k;

            let xl = if idx_l < state_len {
                state[idx_l]
            } else {
                let x = &input[(idx_l - state_len) as usize];
                Complex::new(x.re.as_(), x.im.as_())
            };

            let xr = if idx_r < state_len {
                state[idx_r]
            } else {
                let x = &input[(idx_r - state_len) as usize];
                Complex::new(x.re.as_(), x.im.as_())
            };

            let c = coeff[mid - k];
            acc_re += c * (xl.re + xr.re);
            acc_im += c * (xl.im + xr.im);
        }

        output[i] = Complex::new(acc_re >> bit_shift, acc_im >> bit_shift);
    }

    // 更新状态
    state.copy_from_slice(&input[input.len() - state_len..]
        .iter()
        .map(|x| Complex::new(x.re.as_(), x.im.as_()))
        .collect::<Vec<_>>());

    n_out
}
