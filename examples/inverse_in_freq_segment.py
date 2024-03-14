from typing import Callable

import numpy as np
from numpy.typing import NDArray

from Optimizers import optimize_trust_region
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
import matplotlib.pyplot as plt
from source.jax_plate.ParamTransforms import isotropic_to_full

# http://thermalinfo.ru/svojstva-materialov/metally-i-splavy/plotnost-stali-temperaturnaya-zavisimost/


def make_freq_slice(
    exp_data: NDArray, freq_start: int, freq_end: int, freq_count: int = 100,
):
    """Возвращает NDArray - 0 столбец частоты, 1 столбец амплитуда."""
    start_num = np.where(exp_data[:, 0] > freq_start)[0][0]
    end_num = np.where(exp_data[:, 0] < freq_end)[0][-1]
    required_exp_data = exp_data[start_num:end_num, :]
    k = required_exp_data.shape[0] // freq_count
    if k > 0:
        return required_exp_data[::k, :]
    else:
        return required_exp_data


if __name__ == '__main__':
    print(jax.devices())

    # Steel plate
    rho = 7920.  # [kg/m^3]
    rho = rho*1.025  # [kg/m^3]
    E = 198 * 1e9  # [Pa] # 179736481115.13403
    G = 77 * 1e9  # 58710847964.67647
    nu = E / (2. * G) - 1.  # [1]
    h = 1e-3  # [m]
    D = E * h ** 3 / (12. * (1. - nu ** 2))
    # this value i don't know
    beta = .003  # loss factor, [1]
    accelerometer_params = {'radius': 4e-3, 'mass': 1.0e-3}
    params = jnp.array([2.08501597e+01, 5.30692260e-01, 1.48362224e-02])  # D, nu, beta

    # Инициализация задачи
    p = Problem("_strip_shifted.edp", h, rho, accelerometer_params)
    getAFC = p.getAFCFunction(isotropic_to_full)

    # Формирование референсных данных
    exp_data = np.loadtxt('/home/ivan/src/preparePlateData/res/data_2.txt')
    test_data = make_freq_slice(exp_data=exp_data, freq_start=650, freq_end=750, freq_count=150)
    test_freq, test_afc = test_data[:, 0], test_data[:, 1]
    start_afc = getAFC(jnp.array(test_freq), params)
    ref_afc = jnp.array(test_data)

    loss_function = p.getMSELossFunction(isotropic_to_full, ref_afc[:, 0], ref_afc)
    start_value = loss_function(params)

    plt.plot(ref_afc[:, 0], ref_afc[:, 1], lw=3, label='ref')
    plt.plot(exp_data[:, 0], exp_data[:, 1], label='exp_data')
    plt.plot(test_freq, np.linalg.norm(start_afc, axis=1, ord=2), label='start')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

    opt_result = optimize_trust_region(loss_function, jnp.array(params), N_steps=10, delta_max=0.1, eta=0.15)

    rez_afc = getAFC(jnp.array(test_freq), opt_result.x)
    print(f'{opt_result.x=}')

    fig = plt.figure(figsize=(40, 20))
    ax = fig.subplots(2, 3)

    ax[0][0].plot(ref_afc[:, 0], ref_afc[:, 1], label='ref')
    ax[0][0].plot(test_freq, np.linalg.norm(start_afc, axis=1, ord=2), label='start')
    ax[0][0].plot(test_freq, np.linalg.norm(rez_afc, axis=1, ord=2), label='rez')
    ax[0][0].legend()
    ax[0][0].grid(True)
    ax[0][0].set_yscale('log')
    ax[0][0].set_ylabel('Амплитуда')
    ax[0][0].set_xlabel(r'$\nu$, Гц')

    ax[0][1].plot(opt_result.x_history)
    ax[0][1].grid(True)
    ax[0][1].set_ylabel('Значение искомых величин')
    ax[0][1].set_xlabel('Номер итерации')

    ax[0][2].plot(opt_result.f_history)
    ax[0][2].grid(True)
    ax[0][2].set_ylabel('Значение функционала')
    ax[0][2].set_xlabel('Номер итерации')

    ax[1][0].plot(jnp.linalg.norm(jnp.array(opt_result.grad_history), axis=1))
    ax[1][0].grid(True)
    ax[1][0].set_yscale('log')
    ax[1][0].set_ylabel('Модуль градиента')
    ax[1][0].set_xlabel('Номер итерации')

    ax[1][1].plot(jnp.array(opt_result.grad_history))
    ax[1][1].grid(True)
    ax[1][1].set_ylabel('Градиент')
    ax[1][1].set_xlabel('Номер итерации')


    plt.show()



