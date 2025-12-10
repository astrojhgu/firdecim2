use firdecim2::resample2_complex;
use num::{Complex, Zero};

fn main() {
    let fir_coeffs = vec![
        -26, 0, 28, 0, -31, 0, 36, 0, -44, 0, 53, 0, -65, 0, 79, 0, -97, 0, 117, 0, -140, 0, 167,
        0, -198, 0, 234, 0, -274, 0, 319, 0, -371, 0, 429, 0, -496, 0, 572, 0, -661, 0, 764, 0,
        -887, 0, 1036, 0, -1219, 0, 1454, 0, -1768, 0, 2212, 0, -2897, 0, 4112, 0, -6917, 0, 20848,
        32767, 20848, 0, -6917, 0, 4112, 0, -2897, 0, 2212, 0, -1768, 0, 1454, 0, -1219, 0, 1036,
        0, -887, 0, 764, 0, -661, 0, 572, 0, -496, 0, 429, 0, -371, 0, 319, 0, -274, 0, 234, 0,
        -198, 0, 167, 0, -140, 0, 117, 0, -97, 0, 79, 0, -65, 0, 53, 0, -44, 0, 36, 0, -31, 0, 28,
        0, -26,
    ];
    let mut state: Vec<_> = vec![Complex::zero(); fir_coeffs.len() - 1];
    // let input:Vec<_>=vec![0,0,0,1,0,0].iter().map(|x| Complex::from(*x)).collect();
    // let mut output:Vec<_>=vec![0,0,0,0,0,0].iter().map(|x| Complex::from(*x)).collect();
    // let n=resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    // println!("{}", n);

    // let n=resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    // println!("{}", n);

    let mut input = vec![Complex::<i8>::zero(); 8192];
    input[0] = Complex::new(1, 0);
    //let mut input :Vec<_>=(0..8192).map(|i| Complex::new((127.0*(((i as f64/128.0)*2.0*f64::PI())).sin()) as i8, 0)).collect();
    let mut output = vec![Complex::zero(); input.len() / 2];

    //input[4096]=Complex::<_>::new(1, 1);

    for i in 0..800000 {
        input[1000]=Complex::new((i%127) as i8, 0);
        resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    }
}
