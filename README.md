[![arXiv](https://img.shields.io/badge/arXiv-2306.13105-b31b1b.svg)](https://arxiv.org/abs/2306.13105) [![Kaggle](https://img.shields.io/badge/Kaggle-RadChar-blue?logo=kaggle)](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023) [![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Radar Characterisation Dataset (RadChar)

RadChar is a synthetic radar signal dataset designed to facilitate the development of multi-task learning models. Unlike existing datasets that only provide labels for classification tasks, RadChar provides labels that support both classification and regression tasks in radar signal recognition. This makes it the first multi-task labelled dataset of its kind released to help the research community to advance machine learning for radar signal characterisation.

You can access the most recent paper 📄 here: [https://arxiv.org/abs/2306.13105](https://arxiv.org/abs/2306.13105)

You can also check out our ICASSP 2023 presentation ▶️ here: https://youtu.be/NEsjRYZpJ-8

> Z. Huang, A. Pemasiri, S. Denman, C. Fookes and T. Martin, "Multi-Task Learning For Radar Signal Characterisation," 2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSPW59220.2023.10193318.

## Quick Links

- [RadChar Details](#radchar-details)
- [Example Usage](#example-usage)
- [Mapping of Data Fields](#mapping-of-data-fields)
- [Download Links](#download-links)
- [Citation](#citation)

## RadChar Details

RadChar contains pulsed radar signals at varying signal-to-noise ratios (SNRs) between -20 to 20 dB. This repository provides four variants of the RadChar dataset, which include:

- `RadChar-Tiny` contains 50 thousand radar waveforms;
- `RadChar-Small` contains 500 thousand radar waveforms;
- `RadChar-Baseline` contains 1 million radar waveforms (used in the conference paper); and
- `RadChar-Large` contains 2 million radar waveforms.

Each dataset comprises a total of 5 radar signal types each covering 4 unique signal parameters. The sampling rate used in RadChar is 3.2 MHz. Each waveform in the dataset contains 512 complex, baseband IQ samples.

⚙️ The radar signal types include: 
- Barker codes, up to a code length of 13;
- Polyphase Barker codes, up to a code length of 13;
- Frank codes, up to a code length of 16;
- Linear frequency-modulated (LFM) pulses; and 
- Coherent unmodulated pulse trains. 

⚙️ The radar signal parameters include:
- Number of pulses, sampled between uniform range 2 to 6; 
- Pulse width, sampled between uniform range 10 to 16 µs;
- Pulse repetition interval (PRI), sampled between uniform range 17 to 23 µs; and
- Pulse time delay, sampled between uniform range 1 to 10 µs.

### RadChar Frame

Visualisation of a frame from RadChar:

![RadChar Frame](Sample.png)

## Example Usage

The Python module `h5py` can be used to load the dataset:

```python
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
```

The Python module `matplotlib` can be used to visualise a given radar waveform:

```python
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
```

## Mapping of Data Fields

Each element of the label data in the RadChar dataset corresponds directly to the IQ data, such that the label for a radar waveform at a given index X is associated with the IQ data at the same index X. An example of the label data from the `RadChar-Tiny` dataset is shown below.

```python
array([(    0, 0, 2, 1.40310293e-05, 1.39439383e-06, 2.16004340e-05,  13),
       (    1, 0, 5, 1.06815926e-05, 6.16996193e-06, 2.23167179e-05,  -6),
       (    2, 0, 6, 1.04426709e-05, 6.45071060e-06, 2.02943433e-05,   4),
       ...,
       (49997, 4, 4, 1.32212468e-05, 3.36999220e-06, 2.28439157e-05,   3),
       (49998, 4, 5, 1.06260959e-05, 4.19994184e-06, 2.10669825e-05, -10),
       (49999, 4, 2, 1.14144754e-05, 2.12633509e-06, 1.93383993e-05, -14)],
      dtype=[('index', '<i8'), 
             ('signal_type', '<i8'), 
             ('number_of_pulses', '<i8'), 
             ('pulse_width', '<f8'), 
             ('time_delay', '<f8'), 
             ('pulse_repetition_interval', '<f8'), 
             ('signal_to_noise_ratio', '<i8')])
```

Here, each label contains information about the corresponding radar waveform. The label fields are indexed in the following order:

- `index` - a unique identifier for each waveform
- `signal_type` - signal type following an integer mapping scheme (as shown below) 
- `number_of_pulses` - number of pulses, unitless
- `pulse_width` - pulse width, in seconds
- `time_delay` - pulse time delay, in seconds
- `Pulse pulse_repetition_interval interval` PRI, in seconds
- `signal_to_noise_ratio` - SNR, in dB

Integer mapping of `signal_type`:

```python
signal_type = {'coherent_pulse_train': 0, 
               'barker_code': 1, 
               'polyphase_barker_code': 2,
               'frank_code': 3, 
               'linear_frequency_modulated': 4}
```

## Download Links

The official RadChar dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023). The dataset contains the following variants:
 
- `RadChar-Tiny` - approx. file size of 400 MB
- `RadChar-Small` - approx. file size of 4 GB
- `RadChar-Baseline` - approx. file size of 8 GB
- `RadChar-Large` - approx. file size of 16 GB

> Note, `RadChar-Tiny` is a subset of `RadChar-Small`, while `RadChar-Small` is a subset of `RadChar-Baseline`, etc. It is recommended a train-val-test split should be created from a single RadChar dataset (e.g., `RadChar-Baseline`) to support model development.

## Citation

💡 Please cite both the dataset and the conference paper if you find them helpful for your research. Cheers.

```latex
@inproceedings{huang2023radchar,
  author    = {Zi Huang and Akila Pemasiri and Simon Denman and Clinton Fookes and Terrence Martin},
  title     = {Multi-Task Learning for Radar Signal Characterisation},
  booktitle = {Proceedings of the 2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  year      = {2023},
  pages     = {1--5},
  doi       = {10.1109/ICASSPW59220.2023.10193318},
  keywords  = {Modulation, Radar, Speech recognition, Benchmark testing, Multitasking, Transformers, Task analysis, Multi-task learning, Radio signal recognition, Radar signal characterisation, Automatic modulation classification, Radar dataset, Transformer}
}
```
