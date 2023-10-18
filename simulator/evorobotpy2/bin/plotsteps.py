#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belongs to https://github.com/arthurholtrup/evorobotpy2
   and has been written by Arthur H. Bianchini, arthur.h.bianchini@grad.ufsc.br
   plotsteps.py plots the fitness by the number of steps contained in stat*.npy files

"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

statsumn = 0
statavesum = 0
np.random.seed(1)

if len(sys.argv) == 1:
    cpath = os.getcwd()
    files = os.listdir(cpath)
    print("Plotting data contained in:")
    stats = []
    for f in files:
        if "statS" in f:
            stats.append(f)
    minmax = [[None, None] for _ in range((len(stats)))]
    print(minmax)
    count = 0
    for f in files:
        if "statS" in f:
            print(f)
            fint = f.replace("statS", '')
            fint = fint.replace(".npy", '')
            stat = np.load(f, allow_pickle=True)
            size = np.shape(stat)
            newsize = (int(size[0] / 6), 6)
            if 20 < int(fint) <= 30:
                newsize = (int(size[0]) // 4, 4)
            stat = np.resize(stat, newsize)
            minmax[count][0], minmax[count][1] = min(stat[:, 0]), max(stat[:, 0])
            print(minmax)
            statavesum += 1
            statsumn = statsumn + 1
            count += 1
    
    new_xy = [[None, None] for _ in range(len(stats))]
    ave_esne = []
    ave_es = []
    
    for f in stats:
        fint = f.replace("statS", '')
        fint = fint.replace(".npy", '')
        fint = int(fint)
        stat = np.load(f, allow_pickle=True)
        size = np.shape(stat)
        stat = pd.DataFrame(stat)
        newsize = (int(size[0] / 6), 6)
        if 10 < fint <= 15:
            newsize = (int(size[0]) // 4, 4)
        stat = np.resize(stat, newsize)
        new_x = np.linspace(minmax[fint-1][0], minmax[fint-1][1], 100)
        new_y = np.interp(new_x, stat[:, 0].astype(float), stat[:, 2].astype(float))
        if fint < 11:
            ave_esne.append([new_x, new_y])
        elif fint < 21:
            ave_es.append([new_x, new_y])
    aves = [ave_esne, ave_es]
    label = ["OpenAI-ES-NE","OpenAI-ES"]
    count = 0
    for ave in aves:
        midx = [np.mean([ave[file][0][j] for file in range(len(ave[:]))]) for j in range(100)]
        midy = [np.mean([ave[file][1][j] for file in range(len(ave[:]))]) for j in range(100)]
        plt.plot(midx, midy, label = label[count])
        count += 1
        ci = 1.96 * np.std(midy)/np.sqrt(len(midx))
        plt.fill_between(midx, (midy-ci), (midy+ci), alpha=0.1)
    
    plt.xlabel("Steps")
    plt.ylabel("Generalized fitness")
    plt.title(input("Insert plot title: "))
    plt.legend()
    plt.show()

    if statsumn == 0:
        print("\033[1mERROR: No stat*.npy file found\033[0m")
