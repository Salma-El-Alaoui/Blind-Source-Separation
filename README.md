# Independent Subspace Analysis: Blind Source Separation
Project for the course of Structured Data : Learning, Prediction, Dependency, Testing

## To reproduce our experiments:
Run the following scripts:

- experiments_ecg.py : Script that enables to perform Multidimensional ICA on ecg data (3 channels of ecg, in which we want to separate the baby's ecg' from its mom's', same as in Cardoso's paper (1)) adapted from Cardoso's paper Multidimensional Independent Component Analysis (1). Available implementations of ICA are JADE and FastICA.

- experiments_audio.py : Script that enables to perform ICA, Multidimensional ICA and FastISA on audio data. 
By changing flags, one can perform either:
    - ICA (set flag method to `ica`) on a mixture of two songs (with JADE or FastICA (set flag algorithm to `jade` or `fastICA`)) 
    - MICA on a mixture of tracks (set flag method to `mica`)  on a mixture of two songs (with JADE or FastICA (set flag algorithm to `jade` or `fastICA`)) 
    - fastISA on a a mixture of tracks (set flag method to fastISA)

- experiments_images.py : Script that enables to perform ICA, Multidimensional ICA and FastISA on image data. 
By changing flags, one can perform either:
    - MICA (set flag method to `mica`)  on a mixture of images (with JADE or FastICA (set flag algorithm to `jade` or `fastICA`))
    - fastISA (set flag method to fastISA)

## Source Code:

- projection_utils.py : contains the projection functions necessary to MICA

- data_utils.py : load data and produce mixtures

- HistogramOrientedGradient.py : compute Histogram Oriented Gradient features

- jade.py : implementation of JADE (initially by Cardoso in matlab, translated for Python by Beckers): see http://perso.telecom-paristech.fr/~cardoso/Algo/Jade/jade.py

- fastICA.py : implementation of fastICA and fastISA


