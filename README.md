# ARIEL_Data_Challenge_2019
---

This repository contains all the materials used during the ARIEL Data Challenge. 

This competition was made for the satellite ARIEL which will be launch in 2029 (hope). The mission of this satellite is to caracterize the atmosphere of exoplanets with the hope to find information inside these atmospheres. 


## Data exploration 
---

The first part of the notebook is to explore the data. Each file contains 55 raws with 300 points. Each raw is a wavelength and the point are a mesure of a normalised flux of a synthetic star. The lightcurves represent a transit of different kinds of planets, these lightcurves (in different wavelength) are impacted by stellar spots and gaussian noise (photon noise). 

Different data reduction are presented and the way to store the data in numpy matrices.

## Models 
---

The goal of the challenge was to retrieve the planet parameters even with the different impacts on the lightcurve. To do it, we need to use machine learning. Specially deep learning to model the data.

The file models_dl.py contains deep learning models architectures implemented to fit the data. Two input branches and one output. The output is a list of 55 parameters (float number) representing the ratio of the planet radius on the star radius depending of the wavelength. 

Types of models used: 
  - ANN 
  - LSTM
  - GRU
  - Bidirectional LSTM
  - Attention









