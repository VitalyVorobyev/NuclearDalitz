#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def put_phsp_edges(ax, phsp):
    mABsqLo, mABsqHi = phsp.mABsqRange
    mABsq_space = np.linspace(mABsqLo, mABsqHi, 250)
    mACsqLo, mACsqHi = phsp.mACsqLimAB(mABsq_space)

    ax.plot(mABsq_space, mACsqLo, color='k')
    ax.plot(mABsq_space, mACsqHi, color='k')

def put_resonance(ax, phsp, mass):
    msq = mass**2
    mABsqLo, mABsqHi = phsp.mABsqRange

    if mABsqLo < msq and msq < mABsqHi:
        ax.plot([msq, msq], phsp.mACsqLimAB(msq), color='brown')
        ax.plot(phsp.mACsqLimAB(msq), [msq, msq], color='brown')

def plot_edges_abac(phsp, xlbl=r'$m_{AB}^2$', ylbl=r'$m_{AC}^2$'):
    fig, ax = plt.subplots(figsize=(8, 8))
    put_phsp_edges(ax, phsp)

    ax.set_xlabel(xlbl, fontsize=18)
    ax.set_ylabel(ylbl, fontsize=18)

    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')

    fig.tight_layout()

def dalitz_plot(phsp, sample, xlbl=r'$m_{AB}^2$', ylbl=r'$m_{AC}^2$', show_res=True):
    fig, ax = plt.subplots(figsize=(9, 8))
    put_phsp_edges(ax, phsp)
    if show_res:
        put_resonance(ax, phsp, phsp.mres)

    ax.scatter(sample[:,0], sample[:,1], s=0.3)

    ax.set_xlabel(xlbl, fontsize=18)
    ax.set_ylabel(ylbl, fontsize=18)

    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.axis('equal')

    fig.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/dp.{ext}')


def main(mmo=12.0087, mres=8.01):
    from amplitude import C3Alpha
    from generator import generate_phsp
    phsp = C3Alpha(mmo, mres)
    sample = generate_phsp(phsp, 10**4)
    dalitz_plot(phsp, sample)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(float(sys.argv[1]), float(sys.argv[2]))
    else:
        main()
