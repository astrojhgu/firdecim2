use crossbeam::channel::{Receiver, Sender, bounded};
use firdecim2::firdec_worker::resample2_complex;
use firdecim2::{decim_pipeline::start_decim_pipeline_chain, fir::fir_coeffs};
use lockfree_object_pool::LinearObjectPool;
use num::{Complex, Zero};
use std::hint::black_box;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn main() {
    let patch_len = 8192;
    let fir_coeffs = fir_coeffs();
    let mut state: Vec<Complex<i32>> = vec![Complex::zero(); fir_coeffs.len() - 1];

    let pool_input = Arc::new(LinearObjectPool::new(
        move || vec![Complex::<i8>::zero(); patch_len],
        |_v| {},
    ));

    let pool_output = Arc::new(LinearObjectPool::new(
        move || vec![Complex::<i32>::zero(); patch_len / 2],
        |_v| {},
    ));

    let (send_input, recv_input) =
        bounded::<lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i8>>>>(4);
    let (send_output, recv_output) =
        bounded::<lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i32>>>>(4);

    //let input = recv_input.recv().unwrap();
    //let mut input = vec![Complex::<i8>::zero(); 8192];
    let input=pool_input.pull_owned();
    //let mut input1:Vec<_>=vec![Complex::<i8>::zero(); patch_len];
    //input1.copy_from_slice(&input[..]);
    let mut output = pool_output.pull_owned();
    //let mut output= vec![Complex::zero(); patch_len / 2];
    //let output1:&mut [Complex<i32>]= &mut output;
    let mut output1 = vec![Complex::zero(); patch_len / 2];
    let mut x=0;
    for i in 0..(640_000_000 / patch_len) {
        //input1[1000]=Complex::new((i%127) as i8, 0);
        let input=pool_input.pull_owned();
        resample2_complex(&input, &mut output1[..], &fir_coeffs, &mut state, 0);
        //output.copy_from_slice(&output1[..]);
        black_box(&output1);
    }
    //input[4096]=Complex::<_>::new(1, 1);
    // std::thread::spawn(move || {
    //     for i in 0.. {
    //         let input=recv_input.recv().unwrap();
    //         let mut output = pool_output.pull_owned();
    //         resample2_complex(&input, &mut output, &fir_coeffs, &mut state, 0);
    //         send_output.send(output).unwrap();
    //     }
    // });

    // std::thread::spawn(move || {
    //     for i in 0.. {
    //         recv_output.recv().unwrap();
    //     }
    // });

    // for i in 0..(640_000_000 / patch_len) {
    //     let mut input = pool_input.pull_owned();
    //     input[0]=Complex::new((i%127) as i8, 0);
    //     send_input.send(input).unwrap();
    // }
}
