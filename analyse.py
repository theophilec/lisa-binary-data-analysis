# -*- coding: utf-8 -*-

"""
Analyse module for signal analysis for LISA.

Implements:
- Analyse class.
- Likelihood calculation
- MCMC launching (perhaps should be moved to mcmc.py)
"""


class Analyse:
    import mcmc
    import numpy as np
    import copy
    import noise

    Tobs = 365.25*24*60*60

    def __init__(self, model, data):
        # model signals should already be calculated
        self.noise_psd = Analyse.noise.analytical_noise_psd(model.f)
        # (1.15e-42)*Analyse.np.ones(len(model.A_fourier))
        # valeur de reference d'apres pipeline_antoine.py
        self.model = model
        self.data = data
        self.new_model = Analyse.copy.deepcopy(model)
        self.iFmin = 0
        self.iFmax = len(model.A_fourier) - 1

    @staticmethod
    def full_scalar_product_fourier(model, data, noise_psd, ifmin, ifmax):
        if len(model.A_fourier) != len(data.A_fourier):
            print("Warning : full samples are not the same size...")
            print("Will still try to calculate.")
        # TODO : implement noise by dividing by noise.psd
        # scalar product over A and E channels

        # TODO : check normalization constants
        # NOTE : I've chosen 4.0/T based on papers and Antoine Petiteau.
        s_E = model.A_fourier * Analyse.np.conj(data.A_fourier)/noise_psd
        s_E_1 = s_E[ifmin:ifmax]
        sp_E_channel = Analyse.np.sum(s_E_1)
        sp_E_channel = Analyse.np.real((4./Analyse.Tobs)*sp_E_channel)

        s_A = model.A_fourier * Analyse.np.conj(data.A_fourier)/noise_psd
        s_A_1 = s_A[ifmin:ifmax]
        sp_A_channel = Analyse.np.sum(s_A_1)
        sp_A_channel = Analyse.np.real((4./Analyse.Tobs)*sp_A_channel)

        return sp_E_channel + sp_A_channel

    def basic_scalar_product_fourier(a, b, ifmin, ifmax):
        if len(a) != len(b):
            print("Warning : full samples are not the same size...")
            print("Will still try to calculate.")

        return (4./Analyse.Tobs)*(Analyse.np.sum(a * Analyse.np.conj(b)))[ifmin:ifmax]


    def likelihood(self, next):
        # log likelihood (next if with the jumped model (generation +1))
        if not next:
            return Analyse.full_scalar_product_fourier(self.model, self.data, self.noise_psd, self.iFmin, self.iFmax) -\
             0.5 * Analyse.full_scalar_product_fourier(self.model, self.model, self.noise_psd, self.iFmin, self.iFmax)
        else:
            return Analyse.full_scalar_product_fourier(self.new_model, self.data, self.noise_psd, self.iFmin, self.iFmax) -\
             0.5 * Analyse.full_scalar_product_fourier(self.new_model, self.new_model, self.noise_psd, self.iFmin, self.iFmax)

    @staticmethod
    def basic_likelihood(model, data, noise_psd, ifmin, ifmax):
        return Analyse.full_scalar_product_fourier(model, data, noise_psd, ifmin, ifmax) -\
         0.5 * Analyse.full_scalar_product_fourier(model, model, noise_psd, ifmin, ifmax)

    def launch_mcmc(self, n, startParam, fixedParam, jumpSize, startRand=False):
        self.number_iterations = n
        self.current_iteration = 0
        self.accepted_count = 0

        self.trace_parameters = Analyse.np.zeros((self.number_iterations, 8))
        self.trace_likelihood = Analyse.np.zeros(self.number_iterations)

        self.trace_parameters[0] = startParam
        self.model.set_parameters(startParam)
        self.model.calculate_phi_mod()
        self.model.one_off_calculations()
        self.model.update_all_signals()

        jumpSize = Analyse.np.array(jumpSize)

        print("Beginning MCMC optimization for " + str(n) + " steps.")
        for i in range(1, self.number_iterations):
            self.current_iteration = i
            print("Iteration number " + str(self.current_iteration))
            Analyse.mcmc.iteration(self, jumpSize, fixedParam)
