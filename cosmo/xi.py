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
    j0 = lambda x: [np.sin(ik * x) / (ik * x) if ik !=0 else 1 for ik in k]
    
    Xi = [np.sum(Delta2[:-1] * j0(ir)[:-1] / k[:-1] * dk ) if ir !=0 else 0 for ir in r]

    if mode_coupling:
        A_mc = 1.55
        j1 = lambda x: [np.sin(ik * x) / (ik * x)**2 - np.cos(ik * x) / (ik * x) if ik !=0 else 1 for ik in k]
        Xi1 = [np.sum(Pk[:-1] * j1(ir)[:-1] * k[:-1] * dk) / (2*np.pi**2) if ir!=0 else 0 for ir in r]
        dXi = [(Xi[i+1] - Xi[i]) / (r[i+1] - r[i]) for i in np.arange(len(r[:-1]))]
        dXi = np.append(dXi,dXi[-1])
        Xi = [Xi[i] + A_mc * Xi1[i] * dXi[i] for i in np.arange(len(Xi))]
        
    return r,Xi

def Xi_RSD(r,Xi,b,f,mu):
    '''
    Compute the Redshift Space Distortion correlation function using Hamilton 1992.

    Inputs:

    Outputs:

    '''
    import legendre as L
    
    xi0 = Xi0(Xi,b,f) 
    xi2 = Xi2(r,Xi,b,f) 
    xi4 = Xi4(r,Xi,b,f) 

    Xis = [xi0[i] * L.p0(mu) + xi2[i] * L.p2(mu) + xi4[i] * L.p4(mu) for i in np.arange(len(r))] 
    return r,Xis

def Xi0(Xi,b,f):
    '''
    Compute the redshift space monopole of the correlation function
    '''
    rsd = b**2 + 2./3 * b * f + 1./5 * f**2
    Xi0 = [rsd * iXi for iXi in Xi]
    return Xi0

def Xi2(r,Xi,b,f):
    '''
    Compute the redshift space quadrupole of the correlation function
    '''
    rsd = 4./3 * b *f + 4./7 * f * f
    Xib = [_Xibar(r,Xi,ir) for ir in r] 
    Xi2 = [rsd * (iXi - Xib[i]) for i,iXi in enumerate(Xi)]
    return Xi2

def Xi4(r,Xi,b,f):
    '''
    Compute the redshift space hexadecapole of the correlation function
    '''
    rsd = 8./35 * f * f
    Xib = [_Xibar(r,Xi,ir) for ir in r] 
    Xibb = [_Xibarbar(r,Xi,ir) for ir in r] 
    Xi4 = [rsd * (iXi + 5./2 * Xib[i] - 7./2 * Xibb[i]) for i,iXi in enumerate(Xi)]
    return Xi4

def _Xibar(r,Xi,rmax):
    '''
    Compute the Xibar of Hamilton 1992 (eq 9)
    '''
    if rmax == 0:
        return 0
    Int = [Xi[i] * r[i]**2 * (r[i+1] - r[i]) for i in np.arange(len(r[:-1])) if r[i] < rmax]
    Xibar = 3 * 1./rmax**3 * np.sum(Int)
    return Xibar

def _Xibarbar(r,Xi,rmax):
    '''
    Compute the Xibarbar of Hamilton 1992 (eq 9)
    '''
    if rmax == 0:
        return 0
    Int = [Xi[i] * r[i]**4 * (r[i+1] - r[i]) for i in np.arange(len(r[:-1])) if r[i] < rmax]
    Xibarbar = 5 * 1./rmax**5 * np.sum(Int)
    return Xibarbar
    
