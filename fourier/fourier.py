import numpy as np

def FT_sphere(R,k):
    '''
    Compute the Fourier Transform of a spherical window function of radius R at wavenumber k.

    Input:
         - R : radius of the sphere
         - k : wavenumber at which to compute the FT of the window function
    Return:
         - value of the Fourier Transform of a spherical window function of radius R at wavenumber k
    '''
    ft = 3 / (R*k)**3 * (np.sin(R*k) - R*k*np.cos(R*k))
    return ft
