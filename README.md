# ARIEL_Data_Challenge_2019
---

This repository contains all the materials used during the ARIEL Data Challenge. 

This competition was made for the satellite ARIEL which will be launch in 2029 (hope). The mission of this satellite is to caracterize the atmosphere of exoplanets with the hope to find information inside these atmospheres. 

## Transit

A transit is the phenomenon where we observe the flux of a star and the flux decreases when a planet pass through the line of view. The planet mask a part of the star and this result by a hole in the lightcurve.

<img width="1043" alt="Capture d’écran, le 2021-03-24 à 21 31 32" src="https://user-images.githubusercontent.com/57468886/112405262-53bb0980-8ce8-11eb-9c0c-c232da41e90c.png">
Figure 1: Left: Schematic of an exoplanet transit. The planet passes in-front of the star, obscuring some of the star’s light. This leads to a characteristic dip in stellar flux observed as a function of orbital phase. Middle: Schematic view of transmission spectroscopy whereby some of the stellar light ‘shines through’ the gaseous envelope of a planet. Right: A simulated transmission spectrum of the Ariel mission. Blue are the observed data points and green is a theoretical atmospheric model. Figures courtesy of C. Changeat and adapted from Changeat et al. (2020) Nikolaou et al. (2020).

## Data exploration 
---

The first part of the notebook is to explore the data. Each file contains 55 raws with 300 points. Each raw is a wavelength and the point are a mesure of a normalised flux of a synthetic star. The lightcurves represent a transit of different kinds of planets, these lightcurves (in different wavelength) are impacted by stellar spots and gaussian noise (photon noise). 

Different data reduction are presented and the way to store the data in numpy matrices.

<img width="674" alt="Capture d’écran, le 2021-03-24 à 21 41 44" src="https://user-images.githubusercontent.com/57468886/112406051-d09ab300-8ce9-11eb-9c6a-2951a3d975af.png">
Figure 2: Examples of simulations for two of the 55 wavelength channels, 0.7 μm and 5.6 μm. (a) & (b), stellar surface simulations of a spotty star. Grey line shows the planet transit trajectory. The stellar surface limb brightness varies with wavelength. (c) & (d) Normalised observed flux as the planet transits across the star without stellar photon noise. Blue shows the perfect transit across a spotless star; red shows the transit across a spotty star. (e) & (f) same as (c) & (d) but with stellar photon noise added (Nikolaou et al. 2020).


## Models 
---

The goal of the challenge was to retrieve the planet parameters even with the different impacts on the lightcurve. To do it, we need to use machine learning. Specially deep learning to model the data.

The file models_dl.py contains deep learning models architectures implemented to fit the data. Two input branches and one output. The output is a list of 55 parameters (float number) representing the ratio of the planet radius on the star radius depending of the wavelength. 

Types of models used: 
  - ANN 
  - CNN1D
  - LSTM
  - GRU
  - Bidirectional LSTM
  - Attention


## Leaderboard
---

The leaderboard of the challenge has been published in Nikolaou et al. 2020. The baseline given by the team is of 8726 (87.26% accuracy) so every better result is interested. 13 teams provided better results, the best and winner is 9813. The best result obtained with this notebook is of 9740 (97.40% accuracy) with a deep neural networks. 

<img width="381" alt="Capture d’écran, le 2021-03-24 à 21 23 43" src="https://user-images.githubusercontent.com/57468886/112404718-3df91480-8ce7-11eb-85ba-6335e0e544f0.png">
Table 1: Final leaderboard showing rank & score under Eq.(3) achieved by each of the top-5 entries and the baseline (Nikolaou et a. 2020).

## Results
---


The table below represents the results obtained with 40,000 training files and 10,000 test files. The results were obtained by cross-validation with 10 folds. 

---------------------------------------------------------------------
|  Fold   |   DNN |   DDN2  |  CNN1D  |   LSTM  |   GRU   | Bi-LSTM |
|---------|-------|---------|---------|---------|---------|---------|
|    1    |  9750 |   9720  |   9723  |   9597  |   9489  |   9505  |  
|    2    |  9682 |   9731  |   9717  |   9559  |   9682  |   9556  |
|    3    |  9743 |   9700  |   9711  |   9640  |   9682  |   9667  |
|    4    |  9727 |   9717  |   9730  |   9697  |   9660  |   9597  |
|    5    |  9730 |   9679  |   9746  |   9598  |   9681  |   9509  |
|    6    |  9744 |   9732  |   9683  |   9672  |   9572  |   9263  |
|    7    |  9728 |   9724  |   9725  |   9602  |   9629  |   9620  |
|    8    |  9732 |   9706  |   9758  |   9494  |   9616  |   9564  |
|    9    |  9723 |   9742  |   9734  |   9561  |   9675  |   9617  |
|   10    |  9748 |   9698  |   9736  |   9556  |   9638  |   9504  |
|---------|-------|---------|---------|---------|---------|---------|
| Overall |  9694 |   9740  |   9729  |   9611  |   9659  |   9627  |
---------------------------------------------------------------------


## Acknowledgements 
I would like to thank Dr. Nicolas Cowan and Dr. Nikolaos Nikolaou, the first one for telling me about the challenge and the second one for providing me with the data and the rules of the challenge. 






