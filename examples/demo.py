import jax
import jax.numpy as jnp

from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
from jax_plate.Optimizers import optimize_trust_region, optimize_gd

import matplotlib.pyplot as plt


# Steel plate
rho = 7920. # [kg/m^3]
E = 198*1e9 # [Pa]
G = 77*1e9
nu = E/(2.*G) - 1.# [1]
h = 1e-3 # [m]
D = E*h**3/(12.*(1. - nu**2))
# this value i don't know
beta = .003 # loss factor, [1]

accelerometer_params = {'radius': 4e-3, 'mass': 1.0e-3}

p = Problem("/home/ivan/work/plate_inverse_problem/examples//_strip_shifted.edp", h, rho, accelerometer_params)
get_afc = p.getAFCFunction(isotropic_to_full)
params = jnp.array([D, nu, beta])

freqs = jnp.linspace(0, 1000, 201, endpoint=True)
afc = get_afc(freqs, params)
fig, axs = plot_afc(freqs, afc, label='Test')
perturbed_params = params*jnp.array([0.9, 1.1, 1.1])
loss_function = p.getMSELossFunction(isotropic_to_full, freqs, afc)
loss_and_grad = jax.value_and_grad(loss_function)
loss_and_grad(perturbed_params)