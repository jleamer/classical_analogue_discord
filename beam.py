import numpy as np
from scipy.special import eval_genlaguerre
import pandas as pd

class Beam:
    def __init__(self, pixel_size, num_pixels, I, p1, l1, p2, l2, R, lda):
        self.I = I                                          # intensity of beam
        self.pixel_size = pixel_size                        # side length of pixel in um
        self.num_pixels = num_pixels                        # number of pixels along one side of image
        self.R = R                                          # radius of curvature of beam in um
        self.lda = lda                                      # wavelength of beam in um
        self.length = self.num_pixels * self.pixel_size     # physical length along side of image in um
        self.k = 2 * np.pi / lda                            # wavenumber of beam in inv um
        self.p1 = p1                                        # radial index of beam
        self.l1 = l1                                        # azimuthal index of beam
        self.p2 = p2                                        # radial index of beam
        self.l2 = l2                                        # azimuthal index of beam

        self.E = np.zeros((num_pixels, num_pixels), dtype=complex)

    def make_beam(self, x_p1=0, y_p1=0, x_p2=0, y_p2=0, w1=500, w2=500, psi1=0, psi2=0):
        """
        function for creating beam profile on 2D surface
        :param x_p1:    offset of origin in x-direction in pixels for beam component 1
        :param y_p1:    offset of origin in y-direction in pixels for beam component 1
        :param x_p2:    offset of origin in x-direction in pixels for beam component 2
        :param y_p2:    offset of origin in y-direction in pixels for beam component 2
        :param w:       waist size of beam
        :param psi1:    phase of first beam component
        :param psi2:    phase of second beam component
        :return:
        """
        # Make first beam component
        x_off1, y_off1 = x_p1 * self.pixel_size, y_p1 * self.pixel_size
        x1 = np.linspace(-self.length / 2 + x_off1, self.length / 2 + x_off1, self.num_pixels)
        y1 = np.linspace(-self.length / 2 + y_off1, self.length / 2 + y_off1, self.num_pixels)
        X1, Y1 = np.meshgrid(x1, y1)
        r1 = np.sqrt(X1 ** 2 + Y1 ** 2)
        phi1 = np.angle(X1 + 1j * Y1)
        E1 = (r1 * np.sqrt(2) / w1) ** abs(self.l1) * np.exp(-1j * self.k * r1 ** 2 / (2 * self.R)) * \
            np.exp(-r1 ** 2 / w1 ** 2) * np.exp(-1j * self.l1 * phi1) * \
            eval_genlaguerre(self.p1, self.l1, 2 * r1 ** 2 / w1 ** 2) * \
            np.exp(-1j*psi1)

        # Make second beam component
        x_off2, y_off2 = x_p2 * self.pixel_size, y_p2 * self.pixel_size
        x2 = np.linspace(-self.length / 2 + x_off2, self.length / 2 + x_off2, self.num_pixels)
        y2 = np.linspace(-self.length / 2 + y_off2, self.length / 2 + y_off2, self.num_pixels)
        X2, Y2 = np.meshgrid(x2, y2)
        r2 = np.sqrt(X2 ** 2 + Y2 ** 2)
        phi2 = np.angle(X2 + 1j * Y2)
        E2 = (r2 * np.sqrt(2) / w2) ** abs(self.l2) * np.exp(-1j * self.k * r2 ** 2 / (2 * self.R)) * \
             np.exp(-r2 ** 2 / w2 ** 2) * np.exp(-1j * self.l2 * phi2) * \
             eval_genlaguerre(self.p2, self.l2, 2 * r2 ** 2 / w2 ** 2) * \
             np.exp(-1j*psi2)

        total = E1 + E2
        total /= total.max()
        self.E = np.sqrt(self.I) * total


def get_image(path, I1, I2):
    """
    Function gets the image of the data in path
    :param path: path to file containing experimental data
    :return:     2D array of the data
    """
    df = pd.read_csv(path, sep=';', header=None, skiprows=8).to_numpy()
    intensity = np.array([df[i][0:768].max() for i in range(df[0].size - 1)]).max()
    return df / intensity * np.sqrt(I1 / (I1 + I2))


def psi_get_image(path):
    """
    Function gets the image of the data in path
    :param path: path to file containing experimental data
    :return:     2D array of the data
    """
    df = pd.read_csv(path, sep=';', header=None, skiprows=8).to_numpy()
    intensity = np.array([df[i][0:768].max() for i in range(df[0].size - 1)]).max()
    return df / intensity