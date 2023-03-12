#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@script: Visualise Radar Signal
@datetime: March 13, 2023 
@author: Zi Huang
"""

# Load module
import h5py

# Load the file using h5py
with h5py.File('./RadChar-Tiny.h5', 'r') as f:
    # Print the dataset names
    print(list(f.keys())) # OUT: ['iq', 'labels']
    
    # Get a reference to the dataset
    h5_iqs = f['iq'] 
    h5_labels = f['labels'] 
    
    # Print the shape of the dataset
    print(h5_iqs.shape) # OUT: (50000, 512)
    print(h5_labels.shape) # OUT: (50000,)
    
    # Print the contents of the dataset
    loaded_h5_iqs = h5_iqs[...] # IQ data
    loaded_h5_labels = h5_labels[...] # Label data