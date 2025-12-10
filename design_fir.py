#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
import sys

fs=1.0

def design_halfband_fir(num_taps, window_type='hamming'):
    """
    设计半带低通 FIR 滤波器
    
    Parameters:
        num_taps : int
            FIR 滤波器 tap 数（必须为奇数）
        window_type : str
            窗口类型，如 'hamming', 'hann', 'blackman', 'kaiser'
    
    Returns:
        h : np.ndarray
            半带 FIR 滤波器系数
    """
    if num_taps % 2 == 0:
        raise ValueError("半带滤波器 tap 数必须为奇数")
    
    # 半带滤波器截止频率 fc = 0.25 * fs
    h = firwin(num_taps, cutoff=0.25*fs, window=window_type, fs=fs)
    return h

def plot_frequency_response(h):
    """
    绘制滤波器幅度响应
    """
    w, H = freqz(h, worN=2048)
    plt.figure(figsize=(8,4))
    #plt.plot(w/np.pi, 20*np.log10(np.abs(H)), 'b')
    f = w / np.pi * (0.5)
    plt.plot(f, 20*np.log10(np.abs(H)))
    plt.title("Halfband FIR Filter Frequency Response")
    plt.xlabel("Normalized Frequency [×π rad/sample]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.ylim([-100, 5])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <num_taps> <window_type>")
        print("Example: ./halfband_fir.py 31 hamming")
        sys.exit(1)
    
    num_taps = int(sys.argv[1])
    window_type = sys.argv[2]
    
    h = design_halfband_fir(num_taps, window_type)
    
    plot_frequency_response(h)
    #plt.plot(h)
    plt.plot(h[num_taps//2:],'x')
    plt.show()
    
    print("FIR 系数:")
    print(len(h))
    fir_coeff_int=(h/np.max(h)*((1<<15)-1)).astype('<i4')
    print(f"[{','.join([f'{i}' for i in fir_coeff_int])}]")
    