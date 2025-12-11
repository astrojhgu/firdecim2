use firdecim2::{fir::fir_coeffs, firdec_worker::resample2};
use num::{Complex, Zero};
use std::hint::black_box;

const N_BATCH: usize = 8192;
fn main() {
    let fir_coeffs = fir_coeffs();
    let n_tap_half = fir_coeffs.len();
    let n_tap_full = n_tap_half * 2 - 1;
    let mut state: Vec<_> = vec![0; n_tap_full - 1 + N_BATCH];
    // let input:Vec<_>=vec![0,0,0,1,0,0].iter().map(|x| Complex::from(*x)).collect();
    // let mut output:Vec<_>=vec![0,0,0,0,0,0].iter().map(|x| Complex::from(*x)).collect();
    // let n=resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    // println!("{}", n);

    // let n=resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    // println!("{}", n);

    let mut input = vec![i16::zero(); N_BATCH];
    input[0] = 1;
    //let mut input :Vec<_>=(0..8192).map(|i| Complex::new((127.0*(((i as f64/128.0)*2.0*f64::PI())).sin()) as i8, 0)).collect();
    let mut output = vec![0; input.len() / 2];

    //input[4096]=Complex::<_>::new(1, 1);

    for i in 0..(640_000_000 / 8192) {
        resample2(&input, &mut output, &fir_coeffs, &mut state, 0);
        black_box(&output);
    }
}
