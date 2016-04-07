'''
Set of functions intended to study the correlation function Xi given the power spectrum P(k)
'''
import numpy as np

def Pk2Xi(k,Pk,rmin=0,rmax=200,nbins=200,mode_coupling=False):
    '''
    Transform the 3d power spectrum to the correlation function

    Inputs:
         k  : 1D array containing k values
         Pk : 1D array containing power spectrum value at wavelength k
    
    Optional:
         rmin          : minimum separation at which to compute the correlation function (default 0)
         rmax          : maximum separation at which to compute the correlation function (default 200)
         nbins         : number of bins (default 200)
         mode_coupling : {True, False} include mode coupling form Ross et al. 2011 (eq 19) (default False)

    Outputs:
         r  : 1D array of distances 
         Xi : 1D array of the correlation function
    '''
    r = np.linspace(rmin,rmax,nbins)

    dk = k[1:]-k[0:-1]
    Delta2 = k**3 * Pk / (2*np.pi**2)
    j0 = lambda x: [np.sin(ik * x) / (ik * x) for ik in k]
    
    Xi = [np.sum(Delta2[:-1] * j0(ir)[:-1] / k[:-1] * dk )  for ir in r]

    if mode_coupling:
        A_mc = 1.55
        j1 = lambda x: [np.sin(ik * x) / (ik * x)**2 - np.cos(ik * x) / (ik * x) for ik in k]
        Xi1 = [np.sum(Pk[:-1] * j1(ir)[:-1] * k[:-1] * dk) / (2*np.pi**2) for ir in r]
        dXi = [(Xi[i+1] - Xi[i]) / (r[i+1] - r[i]) for i in np.arange(len(r[:-1]))]
        dXi = np.append(dXi,dXi[-1])
        Xi = [Xi[i] + A_mc * Xi1[i] * dXi[i] for i in np.arange(len(Xi))]
        
    return r,Xi
