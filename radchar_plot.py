#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@script: Visualise Radar Signal
@datetime: March 13, 2023 
@author: Zi Huang
"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt

# Compute time axis
sps = 3.2e6 # Known sampling rate
n = len(loaded_h5_iqs[0])
tmax = n/sps
t = np.linspace(0, tmax, n) # Time horizon
idx = 1000 # Selected radar waveform to be shown 

# Create figure
fig, ax = plt.subplots()
ax.plot(t, np.real(loaded_h5_iqs[idx]), marker='.', markersize=4, 
        color='tab:blue', linestyle='-', linewidth=1.5, 
        alpha=1, label='In-phase') # I component of the IQ signal
ax.plot(t, np.imag(loaded_h5_iqs[idx]), marker='None', markersize=4, 
        color='tab:orange', linestyle='-', linewidth=1.5, 
        alpha=0.75, label='Quadrature') # Q component of the IQ signal

# Using scientific notation for x-axis
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()