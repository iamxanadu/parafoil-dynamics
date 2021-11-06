import matplotlib.pyplot as plt
from numpy.fft import irfft, rfft, rfftfreq
from numpy import linspace, arange, sqrt, interp
from numpy.random import normal

'''
NOTE This is not yet integrated but will be used to generate turbulence for the model
'''


class VonKarmanWind():
    def __init__(self, u20=15, h0=2500, t0=0, v0=5, N=1e3, fs=0.1):
        self.ft2m = 0.3048
        self.u20 = u20

        self.N = int(N)  # Number of samples to draw
        self.fs = fs  # Sampling spatial frequency (1/ft)

        self._update_turb_params(h0)
        self.update(v0, h0, 0)

    def _Psiu(self, w): return sigu**2*2*Lu/pi * 1/(1 + (a*Lu*w)**2)**(5/6)

    def _Psiv(self, w): return sigv**2*2*Lv/pi * \
        (1 + 8/3*(2*a*Lv*w)**2)/(1+(2*a*Lv*w)**2)**(11/6)

    def _Psiw(self, w): return sigw**2*2*Lw/pi * \
        (1 + 8/3*(2*a*Lw*w)**2)/(1+(2*a*Lw*w)**2)**(11/6)

    def _update_turb_params(self, h: float):
        if h < 2000:
            self.sigw = 0.1*self.u20
            self.sigu = self.sigw/(0.177 + 0.000823*h)**0.4
            self.sigv = self.sigw/(0.177 + 0.000823*h)**0.4
            self.Lu = h/(0.177 + 0.000823*h)**1.2
            self.Lv = self.Lu/2
            self.Lw = h/2
        else:
            self.sigu = 5
            self.sigv = 5
            self.sigw = 5
            self.Lu = 2500
            self.Lv = 2500/2
            self.Lw = 2500/2

    def update(self, v: float, h: float, t0: float):
        self.t0 = t0
        '''
        1) Generate three rows of unit variance white gaussian noise samples
        '''
        x = normal(size=(3, self.N))
        '''
        2) Go to the frequency domain. Use real fft because we only care about real arguments.
        '''
        n = rfft(x)
        wf = rfftfreq(self.N, 1/self.fs)
        '''
        3) Make shaping filters using PSDs
        '''
        Gu = sqrt(self.fs*self._Psiu(wf))
        Gv = sqrt(self.fs*self._Psiv(wf))
        Gw = sqrt(self.fs*self._Psiw(wf))
        '''
        4) Transform the noise
        '''
        Yu = Gu*n[0, :]
        Yv = Gv*n[1, :]
        Yw = Gw*n[2, :]
        '''
        5) Go back to the spatial domain
        '''
        yu = irfft(Yu)*self.ft2m
        yv = irfft(Yv)*self.ft2m
        yw = irfft(Yw)*self.ft2m

        self.y = (yu, yv, yw)
        self.t = arange(self.N) / self.fs / v

    def evaluate(self, t: float):
        yu = interp(t - self.t0, self.t, self.y[0])
        yv = interp(t - self.t0, self.t, self.y[1])
        yw = interp(t - self.t0, self.t, self.y[2])

        if t > self.t[-1] + self.t0 or t < self.t0:
            # time out of bounds; return zeros
            return 0, 0, 0
        else:
            return yu, yv, yw

    def plot_von_karman_psd(self):
        wfshift = linspace(-10, 10, 1000)
        plt.figure()
        plt.title("Von Karman Power Spectra")
        plt.plot(wfshift, self._Psiu(wfshift/1000))
        plt.plot(wfshift, self._Psiv(wfshift/1000))
        plt.legend(["Psi_u", "Psi_(v/w)"])
        plt.ylabel("power density (ft/s)^2/(rad/ft)")
        plt.xlabel("frequency (miliradian/ft)")
        plt.show()

    def plot_von_karman_realization(self):
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Von Karman Turbulance Realization")
        plt.plot(self.t, self.y[0])
        plt.ylabel("u (m/s)")
        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.y[1])
        plt.ylabel("v (m/s)")
        plt.subplot(3, 1, 3)
        plt.plot(self.t, self.y[2])
        plt.ylabel("w (m/s)")
        plt.xlabel("time (s)")
        plt.show()
