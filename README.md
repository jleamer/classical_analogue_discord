# Classical Optical Analogue of Quantum Discord
Code for generating results in https://arxiv.org/pdf/2205.00088.pdf

Just a collection of the files and data used to generate the figures in the paper.  Should be possible to just clone the repository and run process_images.py to obtain the images in the paper.  

Some notes:
The beam parameters (like waist size) used for the simulated profiles are presented in process_images.py.  
lambdas_from_mosek.npz contains the values of lambda obtained by solving Eq. 18 for a selection of the experimental_images data.
The experimental data is given as .csv files in experimental_images and its subfolders.  These files note the intensity measured by each pixel of the camera
