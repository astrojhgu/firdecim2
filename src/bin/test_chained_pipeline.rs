use crossbeam::channel::bounded;
use firdecim2::{decim_pipeline::start_decim_pipeline_chain, fir::fir_coeffs};
use lockfree_object_pool::LinearObjectPool;

use num::{Complex, Zero};
use std::{hint::black_box, sync::Arc};

fn main() {
    let fir_coeffs = fir_coeffs();
    let patch_len = 2048;
    let pool = Arc::new(LinearObjectPool::new(
        move || {
            println!("allocated");
            vec![Complex::<i16>::zero(); patch_len]
        },
        |_v| {},
    ));

    let nstages = 6;
    let bit_shifts = vec![0; nstages];
    let (send_input, recv_input) =
        bounded::<lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i16>>>>(64);
    let (_handles, recv_output) = start_decim_pipeline_chain(
        recv_input,
        &fir_coeffs,
        &bit_shifts,
        patch_len,
    );

    std::thread::spawn(move || {
        for _i in 0.. {
            //println!("{}", i);            
            black_box(recv_output.recv().unwrap());
            //println!("got {}", i);
        }
    });

    for _i in 0..(1000_000_000 / patch_len) {
        //for i in 0..800 {
        let input = pool.pull_owned();
        send_input.send(input).unwrap();

        //let output = recv_output.recv().unwrap();
    }
}
//pub fn main() {}
