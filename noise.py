# -*- coding: utf-8 -*-
"""
Noise module for signal analysis for LISA.
Part of the lisa-s3r code base.

Implements:
- Analytical noise PSD (called Sn(f) in literature)
- TODO : correct noise realization.

Integration :
- Called by Analyse object in scalar product.
"""
import numpy as np


def analytical_noise_psd(f):  # frequency np.array
    L0s = 2.5e9/(3.00e8)  # en secondes lumières

    phiL = 2.0*np.pi*L0s*f

    # PSD Optical Noise
    SOpticalNoise = 5.07e-38*f*f

    # PSD Acceleration Noise
    SAccelerationNoise = 6.00e-48*(1+1.e-4/f)/(f*f)

    # PSD of noise for X, Y, Z
    Sn = 16.0*np.sin(phiL)*np.sin(phiL)*(SOpticalNoise +\
                (3.0+np.cos(2.0*phiL))*SAccelerationNoise)

    # Clipping to avoid to lows
    fLim = 0.25/L0s  # 30 MHz
    NoiseMinLowf = 1.e200
    for i in range(len(f)):
        if f[i] < fLim:  # Look for the mininmum
            if Sn[i] < NoiseMinLowf:
                NoiseMinLowf = Sn[i]

        if NoiseMinLowf > Sn[i]:
            Sn[i] = NoiseMinLowf
    Sn[0]=Sn[1]
    return Sn


class Noise: 
    def __init__(self, t, dt):
        self.t = t
        self.dt = dt
        self.f = np.fft.rfftfreq(len(self.t), self.dt)
        Sn = analytical_noise_psd(self.f)
        H = np.sqrt(Sn/(dt**2))  # fonction de transfert
        seed_white_noise = 0.1*np.random.randn(1, len(t))[0]
        seed_white_noise_fourier = np.fft.rfft(seed_white_noise)
        colored_noise_fourier = H * seed_white_noise_fourier
        self.noise = np.fft.irfft(colored_noise_fourier)
