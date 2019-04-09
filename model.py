# -*- coding: utf-8 -*-
"""
Model module for signal analysis for LISA.
Part of the lisa-s3r code base.

Implements:
- Waveform generation in time and frequency domains

Integration :
- Used in Analyse objects.
"""

class Model:

    """ Packages. """
    import numpy as np
    import matplotlib.pyplot as plt
    import gc
    import noise

    """ Class attributes : constants """
    # Length of each arm in light-seconds
    L = 1e9 / 3e8  # change back to 2.5
    # One-year pulsation
    Omega = 2 * np.pi / (60 * 60 * 24 * 365.25)
    # Astronomical unit in light-seconds
    R = 150.0e9 / 3e8

    def __init__(self, t, omega, omega_dot, beta, lamb, h0, phi0, i, psi, is_data=False):
        """ Fundamental vectors and quantities. """
        self.t = Model.np.array(t)
        self.dt = t[1] - t[0]
        self.t_obs = t[-1]
        self.f = None  # frequency array
        self.is_data = is_data

        """ Initializing model parameters. """
        self.omega = omega
        self.omega_dot = omega_dot
        self.beta = beta
        self.lamb = lamb
        self.h0 = h0
        self.phi0 = phi0
        self.i = i
        self.psi = psi

        """ Initializing signals. """
        self.X = None
        self.Y = None
        self.Z = None
        self.A = None
        self.E = None

        self.X1 = None
        self.Y1 = None
        self.Z1 = None
        self.A1 = None
        self.E1 = None

        self.X2 = None
        self.Y2 = None
        self.Z2 = None
        self.A2 = None
        self.E2 = None

        self.X3 = None
        self.Y3 = None
        self.Z3 = None
        self.A3 = None
        self.E3 = None

        self.X4 = None
        self.Y4 = None
        self.Z4 = None
        self.A4 = None
        self.E4 = None

        self.E_fourier = None
        self.A_fourier = None

        self.E1_fourier = None
        self.A1_fourier = None

        self.E2_fourier = None
        self.A2_fourier = None

        self.E3_fourier = None
        self.A3_fourier = None

        self.E4_fourier = None
        self.A4_fourier = None

        """ Initializing intermediate variables. """
        # Calculated by one_off_calculations()
        self.n1 = None
        self.n2 = None
        self.n3 = None

        self.q1 = None
        self.q2 = None
        self.q3 = None

    """ Return parameters. """

    def parameters(self):
        return [self.omega, self.omega_dot, self.beta, self.lamb,
                self.h0, self.phi0, self.i, self.psi]

    def intrinsic_parameters(self):
        return self.omega, self.omega_dot, self.beta, self.lamb

    def extrinsic_parameters(self):
        return self.h0, self.phi0, self.i, self.psi

    """ Update model parameters. """

    def set_parameters(self, p):
        self.omega, self.omega_dot, self.beta, self.lamb, self.h0, self.phi0, \
                    self.i, self.psi = p

    """ Update waveforms. """

    def one_off_calculations(self):
        self.calculate_q_123()
        self.calculate_n_123()

    def calculate_x(self):
        self.X = self.templateX()

    def calculate_y(self):
        self.Y = self.templateY()

    def calculate_z(self):
        self.Z = self.templateZ()

    def update_xyz(self):
        self.calculate_x()
        self.calculate_y()
        self.calculate_z()

    def update_a(self):
        self.A = self.templateA()

    def update_e(self):
        self.E = self.templateE()

    def update_temporal_signals(self):
        self.update_Xs()
        self.update_Ys()
        self.update_Zs()
        self.update_a()
        self.update_e()

        del(self.X1)
        del(self.Y1)
        del(self.Z1)
        del(self.X2)
        del(self.Y2)
        del(self.Z2)
        del(self.X3)
        del(self.Y3)
        del(self.Z3)
        del(self.X4)
        del(self.Y4)
        del(self.Z4)
        del(self.phi_mod)

        if self.is_data:
            self.noise = Model.noise.Noise(self.t, self.dt)
            self.E += self.noise.noise
            self.A += self.noise.noise
            del(self.noise)

        Model.gc.collect()

    def update_all_signals(self):  # updates X, Y, Z and A and E signals
        self.update_temporal_signals()
        self.fourier_transform_signals()

        del(self.E)
        del(self.A)
        Model.gc.collect()

    """ Preliminary calculations. """

    def calculate_q_123(self):
        # Only calculated once : depends only on time (and Omega)
        self.q1 = self.q1_vec()
        self.q2 = self.q2_vec()
        self.q3 = self.q3_vec()

    def q1_vec(self):
        return 1. / (2 * Model.np.sqrt(12)) *\
           Model.np.array([
            Model.np.cos(2 * Model.Omega * self.t - 0) - 3 * Model.np.cos(0),
            Model.np.sin(2 * Model.Omega * self.t - 0) - 3 * Model.np.sin(0),
            -Model.np.sqrt(12) * Model.np.cos(Model.Omega * self.t - 0)
                        ])

    def q2_vec(self):
        return 1. / (2 * Model.np.sqrt(12)) * Model.np.array([
                                            Model.np.cos(2 * Model.Omega * self.t - 2 * Model.np.pi / 3) - 3 * Model.np.cos(2 * Model.np.pi / 3),
                                            Model.np.sin(2 * Model.Omega * self.t - 2 * Model.np.pi / 3) - 3 * Model.np.sin(2 * Model.np.pi / 3),
                                            -Model.np.sqrt(12) * Model.np.cos(Model.Omega * self.t - 2 * Model.np.pi / 3)
                                            ])

    def q3_vec(self):
        return 1. / (2 * Model.np.sqrt(12)) * Model.np.array([
                                            Model.np.cos(2 * Model.Omega * self.t - 4 * Model.np.pi / 3) - 3 * Model.np.cos(4 * Model.np.pi / 3),
                                            Model.np.sin(2 * Model.Omega * self.t - 4 * Model.np.pi / 3) - 3 * Model.np.sin(4 * Model.np.pi / 3),
                                            -Model.np.sqrt(12) * Model.np.cos(Model.Omega * self.t - 4 * Model.np.pi / 3)
                                            ])

    def calculate_n_123(self):  # Only calculated once : idem.

        # Should be unitary
        self.n1 = (self.q2 - self.q3)  # /Model.np.linalg.norm(self.q2-self.q3)
        self.n2 = (self.q3 - self.q1)  # /Model.np.linalg.norm(self.q3-self.q1)
        self.n3 = (self.q1 - self.q2)  # /Model.np.linalg.norm(self.q1-self.q2)

    # Polarisation basis (u seems wrong in the article...)
    # We set u, v, k as a direct orthonormal spherical-like basis,
    # with u = -e_theta, v = e_phi and k = -e_r

    def u_hat(self):
        return -Model.np.array([Model.np.sin(self.beta)*Model.np.cos(self.lamb),
                           Model.np.sin(self.beta)*Model.np.sin(self.lamb),
                           -Model.np.cos(self.beta)])

    def v_hat(self):
        return Model.np.array([Model.np.sin(self.lamb), -Model.np.cos(self.lamb), 0])

    def k_hat(self):
        return - Model.np.array([Model.np.cos(self.beta)*Model.np.cos(self.lamb),
                           Model.np.cos(self.beta)*Model.np.sin(self.lamb), Model.np.sin(self.beta)])

    def u1(self):
        return -0.5 * (Model.np.dot(self.u_hat(), self.n1) ** 2 - \
                       Model.np.dot(self.v_hat(), self.n1) ** 2)

    def u2(self):
        return -0.5 * (Model.np.dot(self.u_hat(), self.n2) ** 2 - \
                       Model.np.dot(self.v_hat(), self.n2) ** 2)

    def u3(self):
        return -0.5 * (Model.np.dot(self.u_hat(), self.n3) ** 2 - \
                       Model.np.dot(self.v_hat(), self.n3) ** 2)

    def v1(self):
        return Model.np.dot(self.u_hat(), self.n1) * Model.np.dot(self.v_hat(), self.n1)

    def v2(self):
        return Model.np.dot(self.u_hat(), self.n2) * Model.np.dot(self.v_hat(), self.n2)

    def v3(self):
        return Model.np.dot(self.u_hat(), self.n3) * Model.np.dot(self.v_hat(), self.n3)

    """ Calculs intermédiaires. """

    """ Phase modulation """

    def calculate_phi_mod(self): # not the same as Jordan ?
        self.phi_mod = self.omega * self.t + 0.5 * self.omega_dot * self.t ** 2 \
        + (self.omega + self.omega_dot * self.t) * Model.R * Model.np.cos(self.beta) * Model.np.cos(Model.Omega * self.t - self.lamb)

    """ Intrinsic parameter templates """

    # We decompose the waveform in the sum of amplitudes and individual waveforms for X, Y and Z,
    # and recombine them to get the components for the A and E channels.
    # h(t) = sum_k (a_k * h_k(t))
    # a_k depend on the extrinsic parameters, h_k(t) depend on the intrinsic parameters
    #
    # We use the long-wavelength approximation
    def calculate_X1(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.u2() - self.u3()) * Model.np.cos(self.phi_mod)

    def calculate_X2(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v2()- self.v3()) * Model.np.cos(self.phi_mod)

    def calculate_X3(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.u2() - self.u3()) * Model.np.sin(self.phi_mod)

    def calculate_X4(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v2()- self.v3()) * Model.np.sin(self.phi_mod)

    def calculate_Y1(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.u3() - self.u1()) * Model.np.cos(self.phi_mod)

    def calculate_Y2(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v3()- self.v1()) * Model.np.cos(self.phi_mod)

    def calculate_Y3(self):
        return 4*(self.omega*Model.L)**2*(self.u3()-self.u1())*Model.np.sin(self.phi_mod)

    def calculate_Y4(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v3() - self.v1()) * Model.np.sin(self.phi_mod)

    def calculate_Z1(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.u1() - self.u2()) * Model.np.cos(self.phi_mod)

    def calculate_Z2(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v1() - self.v2()) * Model.np.cos(self.phi_mod)

    def calculate_Z3(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.u1() - self.u2()) * Model.np.sin(self.phi_mod)

    def calculate_Z4(self):
        return 4 * (self.omega * Model.L) ** 2 * (self.v1() - self.v2()) * Model.np.sin(self.phi_mod)

    def calculate_A1(self):
        return (2 * self.X1 - self.Y1 - self.Z1) / 3.

    def calculate_A2(self):
        return (2 * self.X2 - self.Y2 - self.Z2) / 3.

    def calculate_A3(self):
        return (2 * self.X3 - self.Y3 - self.Z3) / 3.

    def calculate_A4(self):
        return (2 * self.X4 - self.Y4 - self.Z4) / 3.

    def calculate_E1(self):
        return (self.Z1 - self.Y1) / Model.np.sqrt(3)

    def calculate_E2(self):
        return (self.Z2 - self.Y2) / Model.np.sqrt(3)

    def calculate_E3(self):
        return (self.Z3 - self.Y3) / Model.np.sqrt(3)

    def calculate_E4(self):
        return (self.Z4 - self.Y4) / Model.np.sqrt(3)

    def update_Xs(self):
        self.X1 = self.calculate_X1()
        self.X2 = self.calculate_X2()
        self.X3 = self.calculate_X3()
        self.X4 = self.calculate_X4()

    def update_Ys(self):
        self.Y1 = self.calculate_Y1()
        self.Y2 = self.calculate_Y2()
        self.Y3 = self.calculate_Y3()
        self.Y4 = self.calculate_Y4()

    def update_Zs(self):
        self.Z1 = self.calculate_Z1()
        self.Z2 = self.calculate_Z2()
        self.Z3 = self.calculate_Z3()
        self.Z4 = self.calculate_Z4()

    def update_As(self):
        self.A1 = self.calculate_A1()
        self.A2 = self.calculate_A2()
        self.A3 = self.calculate_A3()
        self.A4 = self.calculate_A4()

    def update_Es(self):
        self.E1 = self.calculate_E1()
        self.E2 = self.calculate_E2()
        self.E3 = self.calculate_E3()
        self.E4 = self.calculate_E4()

    """ Extrinsic parameters template """
    """ Does not depend on time, so can be calculated live ! """
    # h(t) = sum_k (a_k * h_k(t))
    # a_k depend on the extrinsic parameters,
    # h_k(t) depend on the intrinsic parameters and time

    def h0t_func(self):  # h0 = A/2 in Cornish/Crowder 2005 -> yes
        return self.h0 * (1 + Model.np.cos(self.i) ** 2)

    def h0x_func(self):  # should be a -1 in factor
        return 2 * self.h0 * Model.np.cos(self.i)

    def a1(self):  # ok for conventions
        return self.h0t_func() * Model.np.cos(self.phi0) * Model.np.cos(2 * self.psi) \
            - self.h0x_func() * Model.np.sin(self.phi0) * Model.np.sin(2 * self.psi)

    def a2(self):  # should be factored -1*
        return self.h0x_func() * Model.np.sin(self.phi0) * Model.np.cos(2 * self.psi) \
            + self.h0t_func() * Model.np.cos(self.phi0) * Model.np.sin(2 * self.psi)

    def a3(self):
        return - (self.h0x_func() * Model.np.cos(self.phi0) * Model.np.sin(2 * self.psi) \
            + self.h0t_func() * Model.np.sin(self.phi0) * Model.np.cos(2 * self.psi))

    def a4(self):
        return self.h0x_func() * Model.np.cos(self.phi0) * Model.np.cos(2 * self.psi) \
            - self.h0t_func() * Model.np.sin(self.phi0) * Model.np.sin(2 * self.psi)

    """ Combination of templates. """
    """ Template combination. """

    def templateX(self):
        return self.a1() * self.X1 + self.a2() * self.X2 + \
           self.a3() * self.X3 + self.a4() * self.X4

    def templateY(self):
        return self.a1() * self.Y1 + self.a2() * self.Y2 + \
               self.a3() * self.Y3 + self.a4() * self.Y4

    def templateZ(self):
        return self.a1() * self.Z1 + self.a2() * self.Z2 + \
               self.a3() * self.Z3 + self.a4() * self.Z4

    def templateA(self):
        return self.a1() * self.calculate_A1() + self.a2() * self.calculate_A2() + \
               self.a3() * self.calculate_A3() + self.a4() * self.calculate_A4()

    def templateE(self):
        return self.a1() * self.calculate_E1() + self.a2() * self.calculate_E2() + \
               self.a3() * self.calculate_E3() + self.a4() * self.calculate_E4()

    """ Fourier transform for signals """

    def update_frequency_array(self):
        self.f = Model.np.fft.rfftfreq(len(self.t), d=self.dt)
        # self.f = self.f[:len(self.E_fourier)]

    def fourier_transform_signals(self):
        # TODO: Is this the best way ? Cutting at right index ?
        # self.E1_fourier = self.dt * Model.np.fft.rfft(self.E1)[:len(self.E1)//2-1]
        # self.A1_fourier = self.dt * Model.np.fft.rfft(self.A1)[:len(self.A1)//2-1]
        # self.E2_fourier = self.dt * Model.np.fft.rfft(self.E2)[:len(self.E2)//2-1]
        # self.A2_fourier = self.dt * Model.np.fft.rfft(self.A2)[:len(self.A2)//2-1]
        # self.E3_fourier = self.dt * Model.np.fft.rfft(self.E3)[:len(self.E3)//2-1]
        # self.A3_fourier = self.dt * Model.np.fft.rfft(self.A3)[:len(self.A3)//2-1]
        # self.E4_fourier = self.dt * Model.np.fft.rfft(self.E4)[:len(self.E4)//2-1]
        # self.A4_fourier = self.dt * Model.np.fft.rfft(self.A4)[:len(self.A4)//2-1]
        self.E_fourier = self.dt * Model.np.fft.rfft(self.E)
        self.A_fourier = self.dt * Model.np.fft.rfft(self.A)
