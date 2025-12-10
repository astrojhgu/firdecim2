use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use crossbeam::channel::{Receiver, Sender};
use lockfree_object_pool::{LinearObjectPool, LinearOwnedReusable};
use num::{Complex, Num, Zero, traits::AsPrimitive};

use crate::firdec_worker::resample2_complex;

pub fn start_decim_pipeline<S, T>(
    recv: Receiver<LinearOwnedReusable<Vec<Complex<S>>>>,
    send: Sender<LinearOwnedReusable<Vec<Complex<T>>>>,
    fir_coeffs: &[T],
    bit_shift: u32,
    patch_len: usize,
    running: Arc<AtomicBool>,
) -> JoinHandle<()>
where
    S: Copy + AsPrimitive<T> + Send + Sync,
    T: 'static
        + Sync
        + Send
        + Copy
        + Zero
        + Num
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Shr<u32, Output = T>,
{
    // Implementation of the decimation pipeline start logic
    let fir_coeffs = fir_coeffs.to_vec();
    std::thread::spawn(move || {
        let ntaps = fir_coeffs.len();
        let state_len = ntaps - 1;
        let mut state = vec![Complex::<T>::zero(); state_len];

        let pool: Arc<LinearObjectPool<Vec<Complex<T>>>> = Arc::new(LinearObjectPool::new(
            move || {
                //eprint!("o");
                vec![Complex::<T>::zero(); patch_len]
            },
            |_v| {},
        ));

        loop {
            
            let mut output = pool.pull_owned();
            
            let input = recv.recv().unwrap();
            resample2_complex(
                &input,
                &mut output[..patch_len / 2],
                &fir_coeffs,
                &mut state,
                bit_shift,
            );

            
            let input = recv.recv().unwrap();
            assert_eq!(input.len(), patch_len);
            resample2_complex(
                &input,
                &mut output[patch_len / 2..],
                &fir_coeffs,
                &mut state,
                bit_shift,
            );

            
            send.send(output).unwrap();
        }
    })
}

pub fn start_decim_pipeline_chain<S, T>(
    recv: Receiver<LinearOwnedReusable<Vec<Complex<S>>>>,
    fir_coeffs: &[T],
    bit_shifts: &[u32],
    patch_len: usize,
    running: Arc<AtomicBool>,
) -> (
    Vec<JoinHandle<()>>,
    Receiver<LinearOwnedReusable<Vec<Complex<T>>>>,
)
where
    S: Copy + AsPrimitive<T> + Send + Sync,
    T: 'static
        + AsPrimitive<T>
        + Sync
        + Send
        + Copy
        + Zero
        + Num
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Shr<u32, Output = T>,
{
    let n_cascades = bit_shifts.len();
    let mut result = Vec::with_capacity(n_cascades);
    let (send1, mut recv1) = crossbeam::channel::bounded::<
        lockfree_object_pool::LinearOwnedReusable<Vec<Complex<T>>>,
    >(4);
    result.push(start_decim_pipeline(
        recv,
        send1,
        fir_coeffs,
        bit_shifts[0],
        patch_len,
        running.clone(),
    ));

    for i in 1..n_cascades {
        let (send1, recv2) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<T>>>,
        >(4);
        let recv = std::mem::replace(&mut recv1, recv2);
        result.push(start_decim_pipeline(
            recv,
            send1,
            fir_coeffs,
            bit_shifts[i],
            patch_len,
            running.clone(),
        ));
    }
    (result, recv1)
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::start_decim_pipeline;
    use crate::fir::fir_coeffs;
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
        let patch_len = 8192;
        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<i8>::zero(); patch_len],
            |_v| {},
        ));

        let (send_input, recv_input) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i8>>>,
        >(4);
        let (send_output, recv_output) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i32>>>,
        >(4);
        let running = Arc::new(AtomicBool::new(true));

        let _handle = start_decim_pipeline(
            recv_input,
            send_output,
            &fir_coeffs,
            0,
            patch_len,
            running.clone(),
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
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i8>>>,
        >(4);
        let nstages = 3;
        let bit_shifts = vec![0; nstages];

        let running = Arc::new(AtomicBool::new(true));

        let (_handles, recv_output) = super::start_decim_pipeline_chain(
            recv_input,
            &fir_coeffs,
            &bit_shifts,
            patch_len,
            running.clone(),
        );

        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<i8>::zero(); patch_len],
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
        let patch_len = 8192;

        let (send_input, recv_input) = crossbeam::channel::bounded::<
            lockfree_object_pool::LinearOwnedReusable<Vec<Complex<i8>>>,
        >(4);
        let nstages = 3;
        let bit_shifts = vec![0; nstages];

        let running = Arc::new(AtomicBool::new(true));

        let (_handles, recv_output) = super::start_decim_pipeline_chain(
            recv_input,
            &fir_coeffs,
            &bit_shifts,
            patch_len,
            running.clone(),
        );

        let pool = Arc::new(LinearObjectPool::new(
            move || vec![Complex::<i8>::zero(); patch_len],
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
