import pathlib
import time
from typing import Callable

import numpy as np
import os
import datetime
from numpy.typing import NDArray

from Optimizers import optimize_trust_region
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
import matplotlib.pyplot as plt
from source.jax_plate.ParamTransforms import isotropic_to_full
from pathlib import Path
from scipy.signal import savgol_filter

# http://thermalinfo.ru/svojstva-materialov/metally-i-splavy/plotnost-stali-temperaturnaya-zavisimost/

NUM_OF_ITERATIONS = 50
SHOW = False

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

def make_freq_slice_near(
    exp_data: NDArray, freqs: list, freq_count: int = 100,
):
    k = freq_count // len(freqs) // 2
    nums = [np.where(exp_data[:, 0] > f)[0][0] for f in freqs]
    ranges = [[n-k if n - k > 0 else 0, n + k if n + k < exp_data.shape[0] else exp_data.shape[0]] for n in nums]
    rez_freq, rez_afc = [], []
    for r in ranges:
        rez_freq += exp_data[r[0]: r[1], 0].tolist()
        rez_afc += exp_data[r[0]: r[1], 1].tolist()
    return np.array([rez_freq, rez_afc]).transpose()


if __name__ == '__main__':
    if not pathlib.Path.exists(Path(__file__).parent.parent / 'rez'):
        os.mkdir(Path(__file__).parent.parent / 'rez')
    rez_path = Path(__file__).parent.parent / 'rez' / str(datetime.datetime.now())
    os.mkdir(rez_path)


    print(jax.devices())

    # Steel plate
    rho = 7920.  # [kg/m^3]
    E = 198 * 1e9  # [Pa] # 179736481115.13403
    nu = 0.28  # [1]
    h = 1e-3  # [m]
    # this value i don't know
    beta = 1.24589327e-02  # loss factor, [1]
    accelerometer_params = {'radius': 4e-3*(1+0.2), 'mass': 1.0e-3}
    params = jnp.array([0.9, 0.5, 0.01])  # E, nu, beta  [0.90619045, 0.53444554, 0.01144685]
    start_params = params * (1. + jnp.array([0.2, 0.05, 0.2]))

    start_data = '\n' + r'$E = $' + "%.5e" % (float(start_params[0]) * 198e9) + '\n' +r'$\nu$ = ' + "%.5e" % float(
        start_params[1]) + '\n' + r'$\beta$ = ' + "%.5e" % float(start_params[2])

    # Инициализация задачи
    p = Problem("_strip_shifted.edp", h, rho, accelerometer_params)
    getAFC = p.getAFCFunction(isotropic_to_full)

    # Формирование референсных данных
    exp_data = np.loadtxt('/home/ivan/src/preparePlateData/res/data_2.txt')
    exp_data = np.array([exp_data[:, 0], savgol_filter(exp_data[:, 1], 10, 2)]).transpose()

    test_data = make_freq_slice_near(exp_data=exp_data, freqs=[675], freq_count=200) # [65, 440, 675, 1260],
    test_freq, test_afc = test_data[:, 0], test_data[:, 1]
    start_afc = getAFC(jnp.array(test_freq), start_params)
    ref_afc = jnp.array(test_data)

    loss_function = p.getMSELossFunction(isotropic_to_full, ref_afc[:, 0], ref_afc)
    start_value = loss_function(start_params)

    plt.scatter(ref_afc[:, 0], ref_afc[:, 1], lw=3, c='b', label='ref')
    plt.plot(exp_data[:, 0], exp_data[:, 1], c='g', label='exp_data')
    plt.scatter(test_freq, np.linalg.norm(start_afc, axis=1, ord=2), c='r', label=f'start: {start_data}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    if SHOW:
        plt.show()
        plt.savefig(str(rez_path / 'start.png'))
    else:
       plt.savefig(str(rez_path / 'start.png'))

    opt_result = optimize_trust_region(loss_function, jnp.array(start_params), N_steps=NUM_OF_ITERATIONS, delta_max=0.1, eta=0.15)


    rez_afc = getAFC(jnp.array(test_freq), opt_result.x)
    print(f'{opt_result.x=}')

    fig = plt.figure(figsize=(40, 20))
    ax = fig.subplots(2, 3)

    ax[0][0].scatter(ref_afc[:, 0], ref_afc[:, 1], lw=3, c='b', label='ref')
    ax[0][0].plot(exp_data[:, 0], exp_data[:, 1], c='g', label='exp_data')
    ax[0][0].scatter(test_freq, np.linalg.norm(start_afc, axis=1, ord=2), c='r', label='start')
    ax[0][0].scatter(test_freq, np.linalg.norm(rez_afc, axis=1, ord=2), c='k', label='rez')
    ax[0][0].legend()
    ax[0][0].grid(True)
    ax[0][0].set_yscale('log')
    ax[0][0].set_ylabel('Амплитуда')
    ax[0][0].set_xlabel(r'$\nu$, Гц')

    ax[0][1].plot(opt_result.x_history)
    ax[0][1].grid(True)
    ax[0][1].set_yscale('log')
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

    jax.config.update('jax_platform_name', 'cpu')
    getAFC = p.getAFCFunction(isotropic_to_full, batch_size=10)
    show_data = make_freq_slice(exp_data=exp_data, freq_start=0, freq_end=2000, freq_count=500)
    show_freq, show_afc = show_data[:, 0], show_data[:, 1]
    start_afc_show = getAFC(jnp.array(show_freq), start_params)
    rez_afc_show = getAFC(jnp.array(show_freq), opt_result.x)
    ax[1][2].plot(exp_data[:, 0], exp_data[:, 1], c='g', label='exp_data')
    ax[1][2].plot(show_freq, np.linalg.norm(start_afc_show, axis=1, ord=2), c='r', lw=4, label='start')
    ax[1][2].plot(show_freq, np.linalg.norm(rez_afc_show, axis=1, ord=2), c='k', label='rez')
    ax[1][2].legend()
    ax[1][2].grid(True)
    ax[1][2].set_yscale('log')
    ax[1][2].set_ylabel('Амплитуда')
    ax[1][2].set_xlabel(r'$\nu$, Гц')

    if SHOW:
        plt.show()
        plt.savefig(str(rez_path / 'end.png'))
    else:
        plt.savefig(str(rez_path / 'end.png'))

    plt.clf()
    plt.cla()
    result_data = '\n' + r'$E = $' + "%.5e" % (float(opt_result.x[0]) * 198e9) + '\n' +r'$\nu$ = ' + "%.5e" % float(
        opt_result.x[1]) + '\n' + r'$\beta$ = ' + "%.5e" % float(opt_result.x[2])
    plt.plot(exp_data[:, 0], exp_data[:, 1], c='g', label='exp_data')
    plt.plot(show_freq, np.linalg.norm(start_afc_show, axis=1, ord=2), c='r', lw=4, label='start')
    plt.plot(show_freq, np.linalg.norm(rez_afc_show, axis=1, ord=2), c='k', label=f'rez: {result_data}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.ylabel('Амплитуда')
    plt.xlabel(r'$\nu$, Гц')

    if SHOW:
        plt.show()
        plt.savefig(str(rez_path / 'afc.png'))
    else:
        plt.savefig(str(rez_path / 'afc.png'))


