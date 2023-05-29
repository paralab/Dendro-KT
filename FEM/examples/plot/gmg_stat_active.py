# Based on this S.O. answer:
# https://stackoverflow.com/a/65179964

import sys

if len(sys.argv) < 2:
    print("Usage: {} <data_file.hdf5>".format(sys.argv[0]), file=sys.stderr)
data_filename = sys.argv[1]


import h5py
import pandas as pd
import matplotlib.animation
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.DataFrame();

def load_and_plot(i):
    matvecs = pd.DataFrame();
    residual = pd.DataFrame();
    groups = []

    try:
        with h5py.File(data_filename,'r') as data:
            groups = list(data.keys())
            for group in groups:
                matvecs = pd.concat([matvecs, pd.Series(data[group]['matvecs']).rename(group)], axis=1)
                residual = pd.concat([residual, pd.Series(data[group]['res_L2']).rename(group)], axis=1)

        plt.cla()

        for group in groups:
            x = matvecs[group]
            y = residual[group]
            plt.plot(x, y, label=y.name)

        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()

    except:
        pass

ani = matplotlib.animation.FuncAnimation(plt.gcf(), load_and_plot, interval=1000)
plt.tight_layout()
plt.show()
