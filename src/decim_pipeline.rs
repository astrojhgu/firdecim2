use std::{
    sync::{
        Arc,
    },
    thread::JoinHandle,
};

use crossbeam::channel::{Receiver, Sender};
use lockfree_object_pool::{LinearObjectPool, LinearOwnedReusable};
use num::{Complex, Zero};

use crate::firdec_worker::resample2;

type DTYPE = i16;

pub fn start_decim_pipeline(
    recv: Receiver<LinearOwnedReusable<Vec<Complex<DTYPE>>>>,
    send: Sender<LinearOwnedReusable<Vec<Complex<DTYPE>>>>,
    fir_coeffs: &[DTYPE],
    bit_shift: u32,
    patch_len: usize,
) -> JoinHandle<()> {
    // Implementation of the decimation pipeline start logic
    let fir_coeffs = fir_coeffs.to_vec();
    std::thread::spawn(move || {
        let ntaps = fir_coeffs.len();
        let state_len = ntaps * 2  - 2 + patch_len; // 2:1 decimation, so input is 2x output
        let mut state = vec![Complex::<DTYPE>::zero(); state_len];
        let state_raw = unsafe {
            std::slice::from_raw_parts_mut(state.as_mut_ptr() as *mut DTYPE, state_len * 2)
        };

        let pool: Arc<LinearObjectPool<Vec<Complex<DTYPE>>>> = Arc::new(LinearObjectPool::new(
            move || {
                //eprint!("o");
                vec![Complex::<DTYPE>::zero(); patch_len]
            },
            |_v| {},
        ));

        loop {
            let mut output = pool.pull_owned();
            let output_raw = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut DTYPE, patch_len * 2)
            };
            if let Ok(input) = recv.recv() {
                let input_raw = unsafe {
                    std::slice::from_raw_parts(input.as_ptr() as *const DTYPE, patch_len * 2)
                };

                resample2(
                    input_raw,
                    &mut output_raw[..patch_len],
                    &fir_coeffs,
                    state_raw,
                    bit_shift,
                );
            } else {
                break;
            }

            if let Ok(input) = recv.recv() {
                let input_raw = unsafe {
                    std::slice::from_raw_parts(input.as_ptr() as *const DTYPE, patch_len * 2)
                };
                assert_eq!(input.len(), patch_len);
                resample2(
                    input_raw,
                    &mut output_raw[patch_len..],
                    &fir_coeffs,
                    state_raw,
                    bit_shift,
                );
            }else{
                break;
            }
            send.send(output).unwrap();
        }
    })
}

pub fn start_decim_pipeline_chain(
    recv: Receiver<LinearOwnedReusable<Vec<Complex<DTYPE>>>>,
    fir_coeffs: &[DTYPE],
    bit_shifts: &[u32],
    patch_len: usize,
) -> (
    Vec<JoinHandle<()>>,
    Receiver<LinearOwnedReusable<Vec<Complex<DTYPE>>>>,
)
{
    let n_cascades = bit_shifts.len();
    let mut result = Vec::with_capacity(n_cascades);
    let (send1, mut recv1) = crossbeam::channel::bounded::<
        lockfree_object_pool::LinearOwnedReusable<Vec<Complex<DTYPE>>>,
    >(32);

    result.push(start_decim_pipeline(
        recv,
        send1,
        fir_coeffs,
        bit_shifts[0],
        patch_len,
    ));

    for i in 1..n_cascades {
        let (send1, recv2) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<DTYPE>>>,
        >(4);
        let recv = std::mem::replace(&mut recv1, recv2);
        result.push(start_decim_pipeline(
            recv,
            send1,
            fir_coeffs,
            bit_shifts[i],
            patch_len,
        ));
    }
    (result, recv1)
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::start_decim_pipeline;
    use crate::{decim_pipeline::DTYPE, fir::fir_coeffs};
    use lockfree_object_pool::LinearObjectPool;
    use num::{
        Complex,
        traits::{FloatConst, Zero},
    };
    use std::{fs::File, sync::Arc};
    use std::{io::Write, sync::atomic::AtomicBool};
    use test::Bencher;

    #[test]
    fn test_decim_pipeline() {
        let fir_coeffs = fir_coeffs();
        let patch_len = 2048;
        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<i16>::zero(); patch_len],
            |_v| {},
        ));

        let (send_input, recv_input) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i16>>>,
        >(4);
        let (send_output, recv_output) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i16>>>,
        >(4);
        let running = Arc::new(AtomicBool::new(true));

        let _handle = start_decim_pipeline(
            recv_input,
            send_output,
            &fir_coeffs,
            0,
            patch_len
        );

        for i in 0..10 {
            let mut input = pool.pull_owned();
            send_input.send(input).unwrap();
            let mut input = pool.pull_owned();
            send_input.send(input).unwrap();
            let output = recv_output.recv().unwrap();
            println!("{}", i);
        }
    }

    #[test]
    fn test_decim_pipeline_chain() {
        let fir_coeffs = fir_coeffs();
        let patch_len = 8192;

        let (send_input, recv_input) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<DTYPE>>>,
        >(4);
        let nstages = 3;
        let bit_shifts = vec![0; nstages];

        let running = Arc::new(AtomicBool::new(true));

        let (_handles, recv_output) = super::start_decim_pipeline_chain(
            recv_input,
            &fir_coeffs,
            &bit_shifts,
            patch_len
        );

        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<DTYPE>::zero(); patch_len],
            |_v| {},
        ));

        for i in 0..10 {
            for _ in 0..(1 << nstages) {
                let mut input = pool.pull_owned();
                send_input.send(input).unwrap();
            }
            let output = recv_output.recv().unwrap();
            println!("{}", i);
        }
    }

    #[bench]
    fn bench_decim_pipeline_chain(b: &mut Bencher) {
        let fir_coeffs = fir_coeffs();
        let patch_len = 2048;

        let (send_input, recv_input) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<DTYPE>>>,
        >(4);
        let nstages = 3;
        let bit_shifts = vec![0; nstages];

        let running = Arc::new(AtomicBool::new(true));

        let (_handles, recv_output) = super::start_decim_pipeline_chain(
            recv_input,
            &fir_coeffs,
            &bit_shifts,
            patch_len,
        );

        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<DTYPE>::zero(); patch_len],
            |_v| {},
        ));

        b.iter(|| {
            println!("bench iteration");
            for i in 0..10 {
                for _ in 0..(1 << nstages) {
                    let mut input = pool.pull_owned();
                    send_input.send(input).unwrap();
                }
                let output = recv_output.recv().unwrap();
            }
        });
    }
}
