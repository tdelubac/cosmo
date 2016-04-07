'''
Set of functions intended to study the correlation function Xi given the power spectrum P(k)
'''
import numpy as np

def Pk2Xi(k,Pk,rmin=0,rmax=200,nbins=200):
    '''
    Transform the 3d power spectrum to the correlation function

    Inputs:
         k  : 1D array containing k values
         Pk : 1D array containing power spectrum value at wavelength k
    
    Optional:
         rmin  : minimum separation at which to compute the correlation function (default 0)
         rmax  : maximum separation at which to compute the correlation function (default 200)
         nbins : number of bins (default 200)

    Outputs:
         r  : 1D array of distances 
         Xi : 1D array of the correlation function
    '''
    r = np.linspace(rmin,rmax,nbins)

    dk = k[1:]-k[0:-1]
    Delta2 = k**3 * Pk / (2*np.pi**2)
    j0 = lambda x: [np.sin(ik * x) / (ik * x) for ik in k]
    
    Xi = [np.sum(Delta2[:-1] * j0(ir)[:-1] / k[:-1] * dk )  for ir in r]
    return r,Xi
