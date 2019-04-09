# -*- coding: utf-8 -*-
"""
MCMC module for signal analysis for LISA.
Part of the lisa-s3r code base.

Implements:
- Metropolis-Hastings algorithm (jumping in canonical parameter space.)
- Printing of steps

Integration :
- Called by Analyse object (launch_mcmc)
"""

import numpy as np
import copy


def jump(analyse, jump_size, fixedParam):
    # vector for jumps
    j = np.zeros(8)
    # vector for new parameters
    new_parameters = np.zeros(8)

    # Jump according to N(0, jump_size)
    # Only jump if FixedParam = 0.
    for i in range(len(jump_size)):
        if not fixedParam[i]:
            j[i] = np.random.randn()*jump_size[i]  # N(0, jump_size^2)
    new_parameters = analyse.model.parameters() + j
    print("Jump in lamba is " + str(j[3]))
    # Jump in magnitude for amplitude.
    # exposant = np.log10(analyse.model.parameters()[4])
    # new_parameters[4] = np.exp(exposant+j[4])
    new_parameters[4] = analyse.model.parameters()[4]  # troubleshooting

    return new_parameters


def rebuildParameters(new_parameters):
    # Puts jumped parameters back into boundary.
    # TODO : is this the right way to proceed ? @theory ?

    p = np.zeros(8)

    # Parameter 0 : omega between 2\pi*fmin et 2\pi*fmax
    fmax = 2*np.pi*1.001e-3
    fmin = 2*np.pi*0.009e-3
    p[0] = max(new_parameters[0], fmin)
    p[0] = min(p[0], fmax)

    # Parameter 1 : OK
    p[1] = new_parameters[1]

    # Parameter 2 et 3 : ecleptical latitude and longitude beta and lambdaa
    beta = new_parameters[2]
    lambdaa = new_parameters[3]

    x = np.cos(beta)*np.cos(lambdaa)
    y = np.cos(beta)*np.sin(lambdaa)
    z = np.sin(beta)

    p[2] = np.arcsin(z)  # in ]-pi/2, pi/2[
    p[3] = np.arctan2(y, x) % (2.*np.pi)  # in [0, 2pi[
    print(p[2])

    # Parameter 4 : amplitude OK
    p[4] = new_parameters[4]

    # Parameter 5 : phase phi0
    p[5] = new_parameters[5] % (2.*np.pi)

    # Parameter 6 : inclination iota in ]-pi, pi[
    new_parameters[6] = new_parameters[6] % (2*np.pi)
    if new_parameters[6] > np.pi:
        p[6] = new_parameters[6] - 2*np.pi
    else:
        p[6] = new_parameters[6]

    # Parameter 7 : polarisation psi
    p[7] = new_parameters[7] % (2*np.pi)

    return p


def acceptRejectJump(analyse, new_parameters):
    # new_parameters must be in boundary already
    # TODO : add seed implementation for reproducible testing ?
    # TODO : make more object oriented ? Yes !
    print("Begin calculating likelihood for MH criterion.")
    # update new model
    analyse.new_model.set_parameters(new_parameters)
    analyse.new_model.calculate_phi_mod()
    analyse.new_model.update_all_signals()

    print("Calculated ok.")
    # calculate difference in inverse log likelihood
    # with new_model and with model
    new_likelihood = analyse.likelihood(True)
    old_likelihood = analyse.likelihood(False)
    print("New likelihood is : " + str(new_likelihood))
    print("Old likelihood is : " + str(old_likelihood))
    print("Data lambda is : " + str(analyse.data.parameters()[3]))
    print("Old model lambda is : " + str(analyse.model.parameters()[3]))
    print("New model lambda is : " + str(analyse.new_model.parameters()[3]))
    print("Data beta is : " + str(analyse.data.parameters()[2]))
    print("Old model beta is : " + str(analyse.model.parameters()[2]))
    print("New model beta is : " + str(analyse.new_model.parameters()[2]))

    delta = new_likelihood - old_likelihood
    ratio = np.exp(delta)
    print("Le delta de likelihood vaut : " + str(delta))
    # Metropolis-Hastings criterion
    a = np.random.rand()
    print("Le a tiré : " + str(a))
    print("Le ratio de likelihood : " + str(ratio))
    print(a < ratio)
    # if np.log(a) < delta
    if a < ratio:
        return True, new_likelihood, old_likelihood
    else:
        return False, new_likelihood, old_likelihood


def iteration(analyse, jump_size, fixedParam):
    # Jump and rebuild parameters
    print("New iteration ")
    p = rebuildParameters(jump(analyse, jump_size, fixedParam))
    print("Rebuilt parameters are : " + str(p))
    accept, new_l, old_l = acceptRejectJump(analyse, p)
    if accept:
        print("Jump accepted.")
        analyse.model = copy.deepcopy(analyse.new_model)
        analyse.accepted_count += 1
        analyse.trace_likelihood[analyse.current_iteration] = new_l
    else:
        print("Jump not accepted.")
        analyse.trace_likelihood[analyse.current_iteration] = old_l
    analyse.trace_parameters[analyse.current_iteration] = analyse.model.parameters()
    print("\n")
