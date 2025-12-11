// use crossbeam::channel::{Receiver, Sender, bounded};
// use firdecim2::{decim_pipeline::start_decim_pipeline_chain, fir::fir_coeffs};
// use lockfree_object_pool::LinearObjectPool;

// use num::{Complex, Zero};
// use std::sync::{
//     Arc,
//     atomic::{AtomicBool, Ordering},
// };

// fn main() {
//     let fir_coeffs = fir_coeffs();
//     let patch_len = 8192;
//     let pool = Arc::new(LinearObjectPool::new(
//         move || {
//             println!("allocated");
//             vec![Complex::<i8>::zero(); patch_len]
//         },
//         |_v| {},
//     ));

//     let nstages = 3;
//     let bit_shifts = vec![0; nstages];
//     let (send_input, recv_input) =
//         bounded::<lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i8>>>>(4);
//     let running = Arc::new(AtomicBool::new(true));

//     let (_handles, recv_output) = start_decim_pipeline_chain(
//         recv_input,
//         &fir_coeffs,
//         &bit_shifts,
//         patch_len,
//         running.clone(),
//     );

//     std::thread::spawn(move || {
//         for i in 0.. {
//             //println!("{}", i);
//             recv_output.recv().unwrap();
//         }
//     });

//     for i in 0..(640_000_000 / patch_len) {
//         //for i in 0..800 {
//         let mut input = pool.pull_owned();
//         send_input.send(input).unwrap();

//         //let output = recv_output.recv().unwrap();
//     }
// }
pub fn main() {}