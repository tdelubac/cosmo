#
# Imports
#
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from copy import copy
from scipy import signal
import legendre as L

class xi:
    def __init__(self,k,Pk):
        self.k = k
        self.Pk = Pk
        self.r = []
        self.mu = []
        self.xi = []
        self.xi_rsd = [] 
        self.xi_lin = []
        self.__xi_coupling = []
        self.mode_coupling = True
        self.A_MC = None
        self.RSD = True
        self.b = None
        self.f = None
        self.__xi_rsd_interp = None

    def set(self,rmin=0,rmax=400,nbinr=400,mumin=0,mumax=1,nbinmu=100,mode_coupling=True,A_MC=1.55,RSD=True,b=1,f=0.3**0.6):
        ''' 
        Set the correlation function given the input k,Pk
        '''
        self.r = np.linspace(rmin,rmax,nbinr)
        self.mode_coupling = mode_coupling
        self.RSD = RSD

        self.__xi_lin()
        self.__mode_coupling()
        self.xi = copy(self.xi_lin)
        
        if self.mode_coupling:
            self.A_MC = A_MC
            self.xi+= A_MC * self.__xi_coupling
            
        if self.RSD:
            self.mu = np.linspace(mumin,mumax,nbinmu)
            self.f = f
            self.__rsd()

        '''Adding bias'''
        self.b=b
        self.xi*= b**2
        self.xi_rsd = np.asarray(self.xi_rsd) * b**2
        return

    def __xi_lin(self):
        '''
        Transform the 3d power spectrum to the linear correlation function
        '''        
        Delta2 = self.k**3 * self.Pk / (2*np.pi**2)
        j0 = lambda x: [np.sin(ik * x) / (ik * x) if ik !=0 else 1 for ik in self.k]
        self.xi_lin = [simps(Delta2 * j0(ir) / self.k,self.k) if ir !=0 else 0 for ir in self.r]
        return

    def __mode_coupling(self):
        '''
        Eq. 19 and 20 of Ross et al . (2011)
        '''
        j1 = lambda x: [np.sin(ik * x) / (ik * x)**2 - np.cos(ik * x) / (ik * x) if ik !=0 else 1 for ik in self.k]
        xi_lin1 = [simps(self.Pk * j1(ir) * self.k,self.k) / (2*np.pi**2) if ir!=0 else 0 for ir in self.r]
        
        dXi = [(self.xi_lin[i+1] - self.xi_lin[i]) / (self.r[i+1] - self.r[i]) for i in np.arange(len(self.r[:-1]))]
        dXi = np.append(dXi,dXi[-1])

        self.__xi_coupling = xi_lin1 * dXi
        return


    def __rsd(self):
        '''
        Compute the Redshift Space Distortion correlation function using Hamilton 1992.
        '''    
        xi0 = self.__Xi0() 
        xi2 = self.__Xi2() 
        xi4 = self.__Xi4() 
        
        self.xi_rsd = [[xi0[i] * L.p0(imu) + xi2[i] * L.p2(imu) + xi4[i] * L.p4(imu) for imu in self.mu] for i in np.arange(len(self.r))]

    def __Xi0(self):
        '''
        Compute the redshift space monopole of the correlation function
        '''
        rsd = (1 + 2./3 * self.f + 1./5 * self.f**2)
        Xi0 = [rsd * iXi for iXi in self.xi]
        return Xi0
    
    def __Xi2(self):
        '''
        Compute the redshift space quadrupole of the correlation function
        '''
        rsd = (4./3 * self.f + 4./7 * self.f**2)
        Xib = [self.__Xibar(ir) for ir in self.r] 
        Xi2 = [rsd * (iXi - Xib[i]) for i,iXi in enumerate(self.xi)]
        return Xi2
    
    def __Xi4(self):
        '''
        Compute the redshift space hexadecapole of the correlation function
        '''
        rsd = (8./35 * self.f**2)
        Xib = [self.__Xibar(ir) for ir in self.r] 
        Xibb = [self.__Xibarbar(ir) for ir in self.r] 
        Xi4 = [rsd * (iXi + 5./2 * Xib[i] - 7./2 * Xibb[i]) for i,iXi in enumerate(self.xi)]
        return Xi4
    
    def __Xibar(self,rmax):
        '''
        Compute the Xibar of Hamilton 1992 (eq 9)
        '''
        if rmax == 0:
            return 0
        Int = [self.xi[i] * self.r[i]**2 for i in np.arange(len(self.r[:-1])) if self.r[i] < rmax]
        Xibar = 3 * 1./rmax**3 * simps(Int,self.r[self.r<rmax])
        return Xibar
    
    def __Xibarbar(self,rmax):
        '''
        Compute the Xibarbar of Hamilton 1992 (eq 9)
        '''
        if rmax == 0:
            return 0
        Int = [self.xi[i] * self.r[i]**4 for i in np.arange(len(self.r[:-1])) if self.r[i] < rmax]
        Xibarbar = 5 * 1./rmax**5 * simps(Int,self.r[self.r<rmax])
        return Xibarbar
    
    def w_theta_rsd(self,theta,z,n,universe):
        '''
        Compute the angular correlation given the redshift space distorded correlation function and the redshift distribution of the measurements. 
        
        Inputs:
             theta : 1D array - angles at which to compute the angular correlation
             z     : 1D array - redshift at which n(z) is sampled
             n     : 1D array - normalized density distribution

        Outputs:
             w : 1D array - angular correlation function w(theta)
        '''       
        interp = interp2d(np.asarray(self.mu),np.asarray(self.r),np.asarray(self.xi_rsd))
        z1,z2 = np.meshgrid(z,z)
    
        w = []
        for i,itheta in enumerate(theta):
            print i # debug
            cost = np.cos(itheta)
            sint = np.sin(itheta)
            dist = np.asarray([[ (universe.Comoving_distance(iiz1)**2 + universe.Comoving_distance(iiz2)**2 - 2*universe.Comoving_distance(iiz1)*universe.Comoving_distance(iiz2)*cost)**(1./2) for (iiz1,iiz2) in zip(iz1,iz2)] for (iz1,iz2) in zip(z1,z2)] )
            zmu = np.asarray([[ np.abs(universe.Comoving_distance(iiz1) - universe.Comoving_distance(iiz2))/iidist if iidist>0 else 0 for (iiz1,iiz2,iidist) in zip(iz1,iz2,idist)] for (iz1,iz2,idist) in zip(z1,z2,dist)])
            w = np.append(w,simps([ n[i1] * simps([n[i2]*interp(zmu[i1][i2],dist[i1][i2])[0] if ( (dist[i1][i2] >= self.r[0]) & (dist[i1][i2] <= self.r[-1]) & (zmu[i1][i2] >= self.mu[0]) & (zmu[i1][i2] <= self.mu[-1]) ) else 0 for i2 in np.arange(len(z))],z) for i1 in np.arange(len(z)) ],z))
        return w

    def w_theta(self,theta,z,n,universe):
        '''
        Compute the angular correlation given the linear correlation function and the redshift distribution of the measurements. 
        
        Inputs:
             theta : 1D array - angles at which to compute the angular correlation
             z     : 1D array - redshift at which n(z) is sampled
             n     : 1D array - normalized density distribution

        Outputs:
             w : 1D array - angular correlation function w(theta)
        '''
        interp = interp1d(np.asarray(self.r),np.asarray(self.xi))
        z1,z2 = np.meshgrid(z,z)
    
        w = []
        for i,itheta in enumerate(theta):
            print i # debug
            cost = np.cos(itheta)
            dist = np.asarray([[ (universe.Comoving_distance(iiz1)**2 + universe.Comoving_distance(iiz2)**2 - 2*universe.Comoving_distance(iiz1)*universe.Comoving_distance(iiz2)*cost)**(1./2) for (iiz1,iiz2) in zip(iz1,iz2)] for (iz1,iz2) in zip(z1,z2)] )
            w = np.append(w,simps([ n[i1] * simps([n[i2]*interp(dist[i1][i2]) if ( (dist[i1][i2] >= self.r[0]) & (dist[i1][i2] <= self.r[-1]) ) else 0 for i2 in np.arange(len(z))],z) for i1 in np.arange(len(z)) ],z))
        return w



    def plot_xi_rsd(self,savefig=''):
        '''
        2D polar plot of the correlation function
        '''
        from matplotlib import pyplot as p
        
        X,Y = np.meshgrid(np.pi/2. -np.arccos(self.mu),self.r)
        fig, ax = p.subplots(subplot_kw=dict(projection='polar'))
        ax.pcolormesh(X, Y,Y*Y*np.asarray(self.xi_rsd))
        ax.pcolormesh(-X, Y,Y*Y*np.asarray(self.xi_rsd))
        ax.pcolormesh(-X+np.pi, Y,Y*Y*np.asarray(self.xi_rsd))
        ax.pcolormesh(X+np.pi, Y,Y*Y*np.asarray(self.xi_rsd))
        
        if savefig=='':
            p.show()
        else:
            p.savefig(savefig)
        return
