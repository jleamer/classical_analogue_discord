import numpy as np
import matplotlib.pyplot as plt
from scipy import special

#import Beam class from beam.py
from beam import Beam, get_image, psi_get_image

plt.rcParams['figure.autolayout'] = True

#######################################################################
#
#   Grab images from experimental data for phi1, phi2 and plot vs
#   simulated data on same figure
#
#######################################################################

# Define paths to use
phi1_path = "experiment_images/phi1/1_180.txt"
phi2_path = "experiment_images/phi2/2_180.txt"

# Get experimental beam profiles
#   Define columns to use - need to be one less than max because data has nans on edge
cols = tuple(_ for _ in np.arange(768))
expm_phi1 = np.genfromtxt(phi1_path, delimiter=';', skip_header=8, usecols=cols)
expm_phi2 = np.genfromtxt(phi2_path, delimiter=';', skip_header=8, usecols=cols)

# Define terms for beams
lda = 798e-3                    # wavelength in um
R = 15e9                        # radius of curvature for beam 1 at z
p1 = 0                          # azimuthal index for beam 1
l1 = 0                          # radial index for beam 1
p2 = 1                          # azimuthal index for beam 2
l2 = 1                          # radial index for beam 2
p3 = 0                          # azimuthal index for beam 3
l3 = 1                          # radial index for beam 3
p4 = 1                          # azimuthal index for beam 4
l4 = 0                          # radial index for beam 4
num_pixels = 768                # number of pixels in image
pixel_size = 5.5                # size of each pixel (on edge) in um

# Define intensities to use
I1 = np.array([27.2, 78.1, 133, 161])
I2 = np.array([136, 130, 137, 90.3])
lambda1 = I1 / (I1 + I2)
lambda2 = I2 / (I1 + I2)

# Create beams - here we only use the last intensities
phi1 = Beam(pixel_size, num_pixels, np.sqrt(lambda1[3]), p1, l1, p2, l2, R, lda)
phi1.make_beam(w1=1100, w2=1100, psi2=3/2*np.pi-0.5, x_p1=5, y_p1=75, x_p2=5, y_p2=75)
phi2 = Beam(pixel_size, num_pixels, np.sqrt(lambda2[3]), p3, l3, p4, l4, R, lda)
phi2.make_beam(w1=850, w2=850, psi1=1/2*np.pi, x_p1=0, y_p1=25, x_p2=25, y_p2=100)

# Plot simulated profiles in a column for fig 1
fig1, ax1 = plt.subplots(2, 1, figsize=[6.4, 4.8])
sim_img1 = ax1[0].imshow(np.abs(phi1.E) ** 2, cmap='viridis')
ax1[0].set_xticks([])
ax1[0].set_yticks([])
sim_img2 = ax1[1].imshow(np.abs(phi2.E) ** 2, cmap='viridis')
ax1[1].set_xticks([])
ax1[1].set_yticks([])
fig1.savefig("sim_beams.png", format='png', dpi=300)

# Plot experimental profiles in a column for fig2
fig2, ax2 = plt.subplots(2, 1, figsize=[6.4, 4.8])
expm_img1 = ax2[0].imshow(expm_phi1, cmap='viridis')
ax2[0].set_xticks([])
ax2[0].set_yticks([])
expm_img2 = ax2[1].imshow(expm_phi2, cmap='viridis')
ax2[1].set_xticks([])
ax2[1].set_yticks([])
fig2.savefig("expm_beams.png", format='png', dpi=300)


#######################################################################
#
#   Grab images from experimental data for psi and plot vs
#   simulated data on same figure for 4 states
#
#######################################################################

# Define paths
phi1_suffixes = ["1_20.txt", "1_60.txt", "1_100.txt", "1_180.txt"]
phi1_paths = ["experiment_images/phi1/" + _ for _ in phi1_suffixes]
phi2_suffixes = ["2_20.txt", "2_60.txt", "2_100.txt", "2_180.txt"]
phi2_paths = ["experiment_images/phi2/" + _ for _ in phi2_suffixes]
psi_suffixes = ["12_20.txt", "12_60.txt", "12_100.txt", "12_180.txt"]
psi_paths = ["experiment_images/psi/" + _ for _ in psi_suffixes]

# Create figures and axes
fig3, ax3 = plt.subplots(4, 3, figsize=[6.4, 7])

# Create list for storing psi values to be used later
psi_list = []

# Load lambda values calculated by mosek
lambda_file = "lamdas_from_mosek.npz"
lambda0s, lambda1s = np.load(lambda_file)['lambda0'], np.load(lambda_file)['lambda1']

# Select the first, third, fifth, and last lambdas (corresponding to the files listed in suffixes)
selected_lambda0s = np.array([lambda0s[0], lambda0s[2], lambda0s[4], lambda0s[8]])
selected_lambda1s = np.array([lambda1s[0], lambda1s[2], lambda1s[4], lambda1s[8]])

# Create lists of experimentally measured intensities
I1 = np.array([27.2, 49.5, 78.1, 107, 133, 160, 159, 164, 161])
I2 = np.array([136, 127, 130, 132, 137, 135, 113, 103, 90.3])

# Calculate the corresponding values of lambda
exp_lambda0 = I1 / (I1 + I2)
exp_lambda1 = I2 / (I1 + I2)

for i in range(4):
    # Load data from files
    phi1 = np.genfromtxt(phi1_paths[i], delimiter=';', skip_header=8, usecols=cols)
    phi2 = np.genfromtxt(phi2_paths[i], delimiter=';', skip_header=8, usecols=cols)
    psi = np.genfromtxt(psi_paths[i], delimiter=';', skip_header=8, usecols=cols)

    # Use phi1 and phi2 with lambda0 and lambda1 to calculate simulated psi
    sim_psi = lambda0s[i] * phi1 + lambda1s[i] * phi2

    psi_list.append(sim_psi)

    # Plot simulated and experimental psis in figs 3 and 4 respectively
    direct_add = exp_lambda0[i] * phi1/phi1.sum() + exp_lambda1[i] * phi2/phi2.sum()
    ax3[i][0].imshow(direct_add, cmap='viridis')
    ax3[i][0].set_xticks([])
    ax3[i][0].set_yticks([])

    expm_psi_img = ax3[i][1].imshow(psi, cmap='viridis')
    ax3[i][1].set_xticks([])
    ax3[i][1].set_yticks([])

    ax3[i][2].imshow(sim_psi, cmap='viridis')
    ax3[i][2].set_xticks([])
    ax3[i][2].set_yticks([])


fig3.savefig("psi_comparison.png", format='png', dpi=300)

#######################################################################
#
#   Generate plot of measured discord vs required discord
#
#######################################################################

# Create linspace for eigenvalues - e3 and e4 will be 0 so no need to define them
e1 = np.linspace(0, 1, 100)
e2 = 1 - e1

# Define r1, r2, r3
r1 = -e1 - e2
r2 = -e1 + e2
r3 = -e1 + e2

# Create list for storing discord values
discord = []
for i in range(100):
    # Calculate r
    r = max(np.abs([r1[i], r2[i], r3[i]]))

    # Calculate sum of x lg x for eigenvalues
    #   Note: scipy's xlogy uses e as the base, but we want to use 2
    eig_sum = special.xlogy(e1[i], e1[i]) / np.log(2)
    eig_sum += special.xlogy(e2[i], e2[i]) / np.log(2)

    # Calculate discord
    discord.append(2 + eig_sum - 0.5 * special.xlogy(1-r, 1-r) / np.log(2) - 0.5 * special.xlogy(1+r, 1+r) / np.log(2))

# Calculate discord using simulated values of lambda0 and lambda1
sim_r1 = -lambda0s - lambda1s
sim_r2 = -lambda0s + lambda1s
sim_r3 = -lambda0s + lambda1s

exp_r1 = -exp_lambda0 - exp_lambda1
exp_r2 = -exp_lambda0 + exp_lambda1
exp_r3 = -exp_lambda0 + exp_lambda1
sim_discord = []
exp_discord = []
for i in range(len(lambda0s)):
    sim_r = max(np.abs([sim_r1[i], sim_r2[i], sim_r3[i]]))
    exp_r = max(np.abs([exp_r1[i], exp_r2[i], exp_r3[i]]))

    sim_eig_sum = special.xlogy(lambda0s[i], lambda0s[i]) / np.log(2)
    sim_eig_sum += special.xlogy(lambda1s[i], lambda1s[i]) / np.log(2)

    sim_discord.append(2 + sim_eig_sum - 0.5 * special.xlogy(1-sim_r, 1-sim_r) / np.log(2) -
                       0.5 * special.xlogy(1+sim_r, 1+sim_r) / np.log(2))

    exp_eig_sum = special.xlogy(exp_lambda0[i], exp_lambda0[i]) / np.log(2)
    exp_eig_sum += special.xlogy(exp_lambda1[i], exp_lambda1[i]) / np.log(2)

    exp_discord.append(2 + exp_eig_sum - 0.5 * special.xlogy(1-exp_r, 1-exp_r) / np.log(2) -
                       0.5 * special.xlogy(1+exp_r, 1+exp_r) / np.log(2))

print("Exp discord: ", exp_discord)
print("Sim discord: ", sim_discord)


fig6, ax6 = plt.subplots(1, 1)
ax6.plot(sim_discord[3:], exp_discord[3:], 'x', ms=8)
ax6.plot([0, 0.1], [0, 0.1], 'k--')
ax6.set_xlabel(r"$\mathcal{D}$", size=14)
ax6.set_ylabel(r"$\mathcal{D}^{\mathrm{Measured}}$", size=14)
ax6.tick_params(labelsize=12)
ax6.set_xlim([-0.002, 0.1])
ax6.set_ylim([-0.002, 0.1])
ax6.grid()
fig6.savefig("measured_vs_setting_discord.png", format='png', dpi=300)
plt.show()
