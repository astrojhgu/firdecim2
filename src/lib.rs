use num::{Complex, Zero};
use num::traits::AsPrimitive;

/// Decimate-by-2 halfband FIR filter
/// input:  &[Complex<S>]
/// output: &mut [Complex<T>]
/// coeff:  &[T]  (halfband FIR, mid tap == 0.5 or similar)
/// state:  filter delay line, len = coeff.len() - 1


/// Optimized safe version: pre-convert input -> T, prebuild window, pre-extract odd coeffs.
/// Not SIMD, but branch-free in inner loop and avoids per-sample conversions.
pub fn resample2_complex<S, T>(
    input: &[Complex<S>],
    output: &mut [Complex<T>],
    coeff: &[T],
    state: &mut [Complex<T>],
    bit_shift: u32,
) -> usize
where
    S: Copy + AsPrimitive<T>,
    T: 'static+Copy
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

    assert!(input.len() % 2 == 0);
    let n_out = input.len() / 2;
    assert!(output.len() >= n_out);

    // 1) Pre-convert input S -> T once into a contiguous buffer
    //    Reuse a buffer allocated on caller if you want zero alloc; here we allocate for clarity.
    let mut input_t: Vec<Complex<T>> = Vec::with_capacity(input.len());
    for x in input.iter() {
        input_t.push(Complex::new(x.re.as_(), x.im.as_()));
    }

    // 2) Build window = [state (len state_len) , input_t (len input.len())] as single Vec
    //    We reuse a Vec so indexes become absolute and branch-free.
    let mut window: Vec<Complex<T>> = Vec::with_capacity(state_len + input_t.len());
    window.extend_from_slice(&state[..]); // copy state into window
    window.extend_from_slice(&input_t[..]); // then the converted input

    // 3) Pre-extract odd taps (offsets and coeffs)
    //    We will store pairs (k,c) where k is odd offset from mid: k=1,3,5...
    let mut odd_offsets: Vec<usize> = Vec::new(); // k values
    let mut odd_coeffs: Vec<T> = Vec::new();
    for k in (1..=mid).step_by(2) {
        odd_offsets.push(k);
        // coeff at index (mid-k) or (mid+k) because symmetric; we choose mid-k
        odd_coeffs.push(coeff[mid - k]);
    }

    // 4) Core loop: for each output i compute pos = 2*i + mid, and use window[pos +/- k] branch-free
    for i in 0..n_out {
        let pos = 2 * i + mid; // index into window
        // center
        let center = window[pos];
        let mut acc_re = coeff[mid] * center.re;
        let mut acc_im = coeff[mid] * center.im;

        // iterate odd taps
        // small unrolling: process in chunks of 4 offsets to help auto-vectorizer
        let mut j = 0usize;
        let n_odd = odd_offsets.len();
        while j + 3 < n_odd {
            let k0 = odd_offsets[j];
            let k1 = odd_offsets[j + 1];
            let k2 = odd_offsets[j + 2];
            let k3 = odd_offsets[j + 3];

            let c0 = odd_coeffs[j];
            let c1 = odd_coeffs[j + 1];
            let c2 = odd_coeffs[j + 2];
            let c3 = odd_coeffs[j + 3];

            let x0l = window[pos - k0]; let x0r = window[pos + k0];
            let x1l = window[pos - k1]; let x1r = window[pos + k1];
            let x2l = window[pos - k2]; let x2r = window[pos + k2];
            let x3l = window[pos - k3]; let x3r = window[pos + k3];

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
        while j < n_odd {
            let k = odd_offsets[j];
            let c = odd_coeffs[j];
            let xl = window[pos - k];
            let xr = window[pos + k];
            acc_re += c * (xl.re + xr.re);
            acc_im += c * (xl.im + xr.im);
            j += 1;
        }

        output[i] = Complex::new(acc_re >> bit_shift, acc_im >> bit_shift);
    }

    // 5) Update state: copy last `state_len` samples from window (which correspond to tail of input)
    let tot = window.len();
    let start = tot - state_len;
    state.copy_from_slice(&window[start..start + state_len]);

    n_out
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