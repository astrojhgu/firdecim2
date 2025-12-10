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
)
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

}


#[cfg(test)]
mod tests{
    use super::resample2_complex;
    use num::Complex;
    use num::traits::Zero;
    use num::traits::FloatConst;


    fn fir_coeffs()->Vec<i32>{
        vec![-26,0,28,0,-31,0,36,0,-44,0,53,0,-65,0,79,0,-97,0,117,0,-140,0,167,0,-198,0,234,0,-274,0,319,0,-371,0,429,0,-496,0,572,0,-661,0,764,0,-887,0,1036,0,-1219,0,1454,0,-1768,0,2212,0,-2897,0,4112,0,-6917,0,20848,32767,20848,0,-6917,0,4112,0,-2897,0,2212,0,-1768,0,1454,0,-1219,0,1036,0,-887,0,764,0,-661,0,572,0,-496,0,429,0,-371,0,319,0,-274,0,234,0,-198,0,167,0,-140,0,117,0,-97,0,79,0,-65,0,53,0,-44,0,36,0,-31,0,28,0,-26]
    }

    #[test]
    fn unit_pulse(){
        let fir_coeffs=fir_coeffs();

        let mut state:Vec<_>=vec![Complex::zero(); fir_coeffs.len()-1];
        let mut input=vec![Complex::<i8>::zero(); fir_coeffs.len()+1];
        input[0]=Complex::new(1,0);
        let mut output=vec![Complex::zero(); input.len()/2];
        resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);        
        fir_coeffs.as_slice().iter().step_by(2).zip(output.iter()).for_each(|(&a,&b)|{
            assert_eq!(a, b.re);
            println!("{} {}", a, b.re);
        });
    }

    #[test]
    fn boundary_test(){
        let fir_coeffs=fir_coeffs();

        let mut state:Vec<Complex<i32>>=vec![Complex::zero(); fir_coeffs.len()-1];
        let input :Vec<_>=(0..8192).map(|i| Complex::new((127.0*(((i as f64/128.0)*2.0*f64::PI())).sin()) as i8, 0)).collect();
        let mut output1=vec![Complex::zero(); input.len()/2];

        resample2_complex(&input, &mut output1, &fir_coeffs, &mut state, 0);
        let mut state:Vec<Complex<i32>>=vec![Complex::zero(); fir_coeffs.len()-1];

        let mut output2=vec![Complex::zero(); input.len()/2];
        resample2_complex(&input[0..4000], &mut output2[0..2000], &fir_coeffs, &mut state, 0);

        resample2_complex(&input[4000..], &mut output2[2000..], &fir_coeffs, &mut state, 0);
        output1.iter().zip(output2.iter()).for_each(|(a,b)|{
            assert_eq!(a,b);
            println!("{} {}", a,b);
        });

    }
}