import os

import jax
import jax.numpy as jnp

import ParamTransforms
from time import perf_counter
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
from jax_plate.Optimizers import optimize_trust_region, optimize_gd

import matplotlib.pyplot as plt

from source.jax_plate.ParamTransforms import isotropic_to_full

print(jax.devices())

# Steel plate
rho = 7920. # [kg/m^3]
E = 198*1e9 # [Pa] # 179736481115.13403
G = 77*1e9 # 58710847964.67647
nu = E/(2.*G) - 1.# [1]
h = 1e-3 # [m]
D = E*h**3/(12.*(1. - nu**2))
# this value i don't know
beta = .003 # loss factor, [1]

accelerometer_params = {'radius': 4e-3, 'mass': 1.0e-3}

p = Problem("_strip_shifted.edp", h, rho, accelerometer_params)
get_afc = p.getAFCFunction(isotropic_to_full)
params = jnp.array([D, nu, beta])

start_params = params*(jnp.array([0.2, 0.05, 99.]) + 1.)
#start_params= jnp.array([2.08501597e+01, 5.30692260e-01, 1.48362224e-02])
getAFC = p.getAFCFunction(isotropic_to_full)

exp_data = np.loadtxt('/home/ivan/src/preparePlateData/res/data_2.txt')

plt.plot(exp_data[:, 0])
plt.xlabel('Номер частоты')
plt.ylabel(r'$\nu$, Гц')
plt.grid(True)
plt.show()


TOTAL_NUMBER = 100
k = exp_data.shape[0] // TOTAL_NUMBER
freqs, afc = exp_data[::k, 0], exp_data[::k, 1]
start_afc = getAFC(jnp.array(freqs), start_params)
ref_afc = jnp.array([[f, a] for f, a in zip(freqs, afc)])

loss_function = p.getMSELossFunction(isotropic_to_full, freqs, ref_afc)

rez1_afc = getAFC(exp_data[::exp_data.shape[0] // 200, 0], jnp.array([2.08501597e+01, 5.30692260e-01, 1.48362224e-02]))
rez2_afc = getAFC(exp_data[::exp_data.shape[0] // 200, 0], jnp.array([2.33569749e+01, 5.77379402e-01, 1.33975109e-02]))
plt.plot(exp_data[::exp_data.shape[0] // 200, 0], np.linalg.norm(rez1_afc, axis=1, ord=2), label='rez1')
plt.plot(exp_data[::exp_data.shape[0] // 200, 0], np.linalg.norm(rez2_afc, axis=1, ord=2), label='rez2')
plt.plot(exp_data[:, 0], exp_data[:, 1], label='exp_data')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

# plt.plot(ref_afc[:, 0], ref_afc[:, 1], label='ref')
# plt.plot(freqs, np.linalg.norm(start_afc, axis=1, ord=2), label='start')
# plt.legend()
# plt.yscale('log')
# plt.show()
#
# t = perf_counter()
# opt_result = optimize_trust_region(loss_function, jnp.array(start_params), N_steps=10, delta_max=0.1, eta=0.15)
# dt = perf_counter() - t
#
# rez_afc = getAFC(jnp.array(freqs), opt_result.x)
#
# plt.plot(opt_result.grad_history)
# plt.yscale('log')
# plt.show()
#
# # plt.plot(jnp.linalg.norm(opt_result.grad_history))
# # plt.show()
#
#
#
#
# plt.plot(freqs, np.linalg.norm(start_afc, axis=1, ord=2), label='start')
# plt.plot(freqs, np.linalg.norm(rez_afc, axis=1, ord=2), label='rez')
# plt.plot(ref_afc[:, 0], ref_afc[:, 1], label='ref')
# plt.legend()
# plt.yscale('log')
# plt.show()
#
# np.savetxt(
#     'start',
#     np.array(
#         [[f, a] for f, a in zip(freqs, np.linalg.norm(start_afc, axis=1, ord=2))]
#     )
# )
# np.savetxt(
#     'rez',
#     np.array(
#         [[f, a] for f, a in zip(freqs, np.linalg.norm(rez_afc, axis=1, ord=2))]
#     )
# )
#
# print('Elapsed time,', dt, 's')
# print(opt_result.x)
#
# os._exit(0)