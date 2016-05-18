#
# Imports
#
from __future__ import print_function
import numpy as np
import numbers
import os
import fileinput
import random
import mcint
import itertools
from ..sky import sky 
from ..fourier import fourier
from scipy import integrate
from scipy import misc
from scipy import interpolate
from numpy import fft
from matplotlib import pyplot as p
from astropy.io import ascii
#
# Class
#
class Universe:
    def __init__(self):
        # Cosmo parameters (Planck + WP 2014)
        self.Omega_m_h2 = 0.14195
        self.Omega_b_h2 = 0.02205
        self.Omega_l = 0.685
        self.h = 0.673
        self.sigma8 = 0.829
        self.n_s = 0.96
        # Other science parameters
        self.Omega_r = 8.24e-5
        self.c = 299792.458 #(km/s) 
        self.Theta27 = 1.01 # Temperature CMB in unit 2.7K
        # class parameters
        self.exact_distances_threshold = 0.01
        self.camb_path = '/Users/tdelubac/Work/Log/camb/'
        # Booleans
        self.exact_distances = False
        # Pk
        self.Pk = []
        self.kmin = 0
        self.kmax = 0
        self.window_wave = []
        self.window_func = []
#--------------------------------
#
# Dealing with class variables
#
#--------------------------------
    def Get_param(self,z=0):
        '''
        Return dictionary with parameters value at redshift z
        '''
        h = self.h
        Omh2 = self.Omega_m_h2 * (1+z)**3
        Obh2 = self.Omega_b_h2 * (1+z)**3
        Or = self.Omega_r * (1+z)**4 * self.h**2 / h**2
        Ol = self.Omega_l * self.h**2 / h**2
        Theta27 = self.Theta27 * (1+z)**4
        return {'Omega_m_h2':Omh2, 'Omega_b_h2':Obh2, 'Omega_l':Ol, 'Omega_r':Or, 'h':h, 'c':self.c, 'Theta27':Theta27, 'sigma8':self.sigma8, 'n_s':self.n_s}

    def Get_param_unit(self):
        '''
        Return dictionary with parameters unit
        '''
        return {'Omega_m_h2':'none', 'Omega_b_h2':'none', 'Omega_l':'none', 'Omega_r':'none', 'h':'none', 'c':'km/s', 'Theta27':'2.7K', 'sigma8':'none', 'n_s':'none'}

    def Set_param(self, Omega_m_h2=False, Omega_b_h2=False, Omega_l=False, Omega_r=False, h=False, c=False, Theta27=False, sigma8=False, n_s=False):
        '''
        Set parameters
        '''
        if Omega_m_h2 != False:
            self.Omega_m_h2 = Omega_m_h2
        if Omega_b_h2 != False:
            self.Omega_b_h2 = Omega_b_h2
        if Omega_l != False:
            self.Omega_l = Omega_l
        if Omega_r != False:
            self.Omega_r = Omega_r
        if h != False:
            self.h = h
        if c != False:
            self.c = c
        if n_s != False:
            self.n_s = n_s
        if sigma8 != False:
            self.sigma8 = sigma8
        if Theta27 != False:
            self.Theta27 = Theta27
        if (self.exact_distances==False) & (np.abs(self.Omega_k())<self.exact_distances_threshold):
            print('### INFO ### Sin_comoving_distance : Omega_k =',self.Omega_k(),', will assume flatness for distance estimations (set exact_distances to true for exact estimations)')
        return

    def Get_distance_unit(self):
        '''
        Return distance unit (Mpc/h)
        '''
        return 'Mpc/h'

    def Set_exact_disctances(self,bool):
        '''
        Set exact_distances boolean. If true will not assume flatness for distance estimations even if very close to flatness
        '''
        self.exact_distances = bool
        return

    def Get_camb_path(self):
        '''
        Print current path to Camb directory
        '''
        return self.camb_path
        
    def Set_camb_path(self,path):
        '''
        Set path to Camb directory
        '''
        self.camb_path = path
        return
#--------------------------------
#
# Get basic information
#
#--------------------------------
    def Omega_t(self,z=0):
        '''
        Return Omega total at redshift z (default z=0)
        '''
        params = self.Get_param(z)
        return params['Omega_m_h2']/params['h']**2 + params['Omega_l'] + params['Omega_r']

    def Omega_k(self,z=0):
        '''
        Return Omega curvature at redshift z (default z=0)
        '''
        return 1 - self.Omega_t(z)
#--------------------------------
#
# Distances
#
#--------------------------------
    def H(self,z):
        '''
        Compute Hubble parameter at redshift z
        '''
        Ot = self.Omega_m_h2/self.h**2 + self.Omega_l + self.Omega_r
        Ok = 1 - Ot
        hubble = 100*np.sqrt(self.Omega_r*(1+z)**4 + self.Omega_m_h2/self.h**2*(1+z)**3 + Ok*(1+z)**2 + self.Omega_l)
        return hubble

    def Comoving_distance(self,z):
        '''
        Compute comoving distance to redshift z if z is a number or between z[0] and z[1] if z is a list. 
        '''
        if isinstance(z,numbers.Number):
            if z<=0:
                return 0
            z_0,z_1 = 0,z
        else:
            z_0 = min(z)
            z_1 = max(z)
        if z_0 == z_1:
            return 0
        I = integrate.quad(lambda x: 1./self.H(x),z_0,z_1)
        if I[0]>0:
            if I[1]/I[0]>0.01:
                print('### WARNING ### Comoving_distance : estimate has error of',I[1]/I[0]*100,'%' )
        Chi = I[0]*self.c
        return Chi

    def Tranverse_comoving_distance(self,z):
        '''
        Compute the tranverse comoving distance (chi, sin(chi) or sinh(chi) depending on Omega_k)
        '''
        Ok = self.Omega_k()
        d_H = self.c/100
        if (Ok == 0) | ( (np.abs(Ok)<self.exact_distances_threshold) & (self.exact_distances==False) ):
            return self.Comoving_distance(z)
        elif Ok>0:
            sqrtOk = np.sqrt(Ok)
            return d_H/sqrtOk*np.sin(self.Comoving_distance(z)*sqrtOk/d_H)
        elif Ok<0:
            sqrtOk = np.sqrt(np.abs(Ok))
            return d_H/sqrtOk*np.sinh(self.Comoving_distance(z)*sqrtOk/d_H)
        
    def Angular_distance(self,z):
        '''
        Compute angular distance to redshift z
        '''
        if z<=0:
            return 0
        else:
            return self.Tranverse_comoving_distance(z)/(1+z)

    def Luminosity_distance(self,z):
        '''
        Compute luminosity distance to redshift z
        '''
        if z<=0:
            return 0
        else:
            return self.Tranverse_comoving_distance(z)*(1+z)
#--------------------------------
#
# Surface
#
#--------------------------------
    def Comoving_surface(self,z,ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
        '''
        Compute the comoving surface at given redshift
        z : value at which to compute comoving surface
        ra = [ra_min, ra_max] : 2-dimensional array containing RA bounds (default radians) 
        dec = [dec_min, dec_max] : 2-dimensional array containing Dec bounds (default radians) 
        degrees = {True, False} : if True uses degrees, else radians
        '''
        ang = sky.Solid_angle(ra,dec,degrees)
        if degrees == True:
            ang*=np.pi**2/180**2
        return ( (1+z)*self.Angular_distance(z) )**2 * ang
#--------------------------------
#
# Volume
#
#--------------------------------
    def Comoving_volume(self,z,ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
        '''
        Compute the comoving volume given a redshift range and angles on the sky (similar as Comoving_volume)
        z = [z_min, z_max] : 2-dimensional array containing redshift bounds
        ra = [ra_min, ra_max] : 2-dimensional array containing RA bounds (default radians) 
        dec = [dec_min, dec_max] : 2-dimensional array containing Dec bounds (default radians) 
        degrees = {True, False} : if True uses degrees, else radians
        '''
        I = integrate.quad(lambda x: self.Comoving_surface(x,ra,dec,degrees)*misc.derivative(self.Comoving_distance,x,dx=1e-6),z[0],z[1])
        return I[0]
#--------------------------------
#
# Pk
#
#--------------------------------
    def Pk_Camb(self,z=0):
        '''
        Compute Pk at redshift z (default z=0) using Camb with cosmological parameters of the class.
        '''
        prefix = 'myuniverse_'
        inifile = prefix+'camb.ini'
        outfile = prefix+'matterpower.dat'
        os.system('cp /Users/tdelubac/Work/ELG/Macro/myuniverse/data/camb.ini /Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini')
        params = self.Get_param(z)
        for line in fileinput.input('/Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini',inplace=True):
            print(line.replace('ombh2          = 0.0226','ombh2          = '+'{0:.4f}'.format(params['Omega_b_h2'])), end='')
        for line in fileinput.input('/Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini',inplace=True):
            print(line.replace('omch2          = 0.112','omch2          = '+'{0:.4f}'.format(params['Omega_m_h2']-params['Omega_b_h2'])), end='')
        for line in fileinput.input('/Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini',inplace=True):
            print(line.replace('omk            = 0','omk            = '+'{0:.4f}'.format(self.Omega_k(z))), end='')
        for line in fileinput.input('/Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini',inplace=True):
            print(line.replace('hubble         = 70','hubble         = '+'{0:.4f}'.format(params['h']*100)), end='')
        os.system('cd '+self.camb_path)
        os.system('mv /Users/tdelubac/Work/ELG/Macro/myuniverse/data/myuniverse_camb.ini '+os.path.join(self.camb_path,inifile))
        print('### INFO ### Pk_Camb : Invoking camb')
        os.system(os.path.join('./camb')+' '+os.path.join(self.camb_path,inifile))
        os.system('cd '+self.camb_path)
        camb_results = ascii.read(os.path.join(self.camb_path,outfile))
        k = camb_results['col1'].data
        Pk = camb_results['col2'].data
        return (k,Pk)

    def Pk_growth(self,z=0,sig_v=4.48,damping=True):
        '''
        Compute Pk at redshift z (default z=0) using Pk from Camb at z=0 with cosmological parameters of the class plus linear growth factor and damping.
        
        Input:
        - z (=0)            : Redshift at which Pk is evaluated
        - damping (=True)   : Whether to include streaming damping
        - sig_v (=4.48)     : Value of the streaming scale. Default is 4.47 Mpc = 3 Mpc/h for h = 0.67 which is the value of Anderson et al. 2014

        Output:
        - [k,Pk]            : In (Mpc/h)^3
        '''
        Pk0 = self.Pk_Camb()
        D = self.Linear_growth(z)
        if damping:
            Pk = [D**2 * iPk * np.e**(-(ik*sig_v)**2) for ik,iPk in zip(Pk0[0],Pk0[1])]
        else:
            Pk = [D**2 * iPk for iPk in Pk0[1]]
        return [Pk0[0],Pk]

    def Linear_growth(self,z=0):
        '''
        Compute the linear growth D(z) (default z=0) normalized to D(z=0) for Lambda cosmologies following Heath 1977 (see Percival 2005 eq. 15)
        '''
        I0 = integrate.quad(lambda x: (1+x)/(self.H(x)/(100))**3,0,3000)
        I = integrate.quad(lambda x: (1+x)/(self.H(x)/(100))**3,z,3000)
        if I[1]/I[0]>0.01:
            print('### WARNING ### Linear_growth : estimate has error of',I[1]/I[0]*100,'%' )
        D0 = 5./2*self.Omega_m_h2/self.h**2 * I0[0]
        D = 5./2*self.Omega_m_h2/self.h**2 * self.H(z)/(100) * I[0]
        return D/D0
                  
    def Pk_EH(self,k,z=0):
        '''
        Compute the theoretical power spectrum using the Eisenstein & Hu transfer function 
        Assume flatness through computation of delta_h (Eisenstein & Hu 1998 eq: A3)
             k : array of values at which to compute P(k).
        optional:
             z : redshift at which to compute P(k). Default 0. 
        '''
        import eisenstein_hu as eh
        k_mpc = k*self.h
        Pk = eh.Pk(self,k_mpc,z)
        return Pk

    def Pk_Jeff(self,k):
        n = 1
        gamma = 0.21
        sigma8=0.96
        keff=(0.172+0.011*(np.log(gamma/0.34))**2)*1.
        q=keff/(self.h*gamma)
        tk=np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25)
        sigma8squn=keff**(3+n)*tk**2
        q=k/(self.h*gamma)
        tk=np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25)
        delsq=k**n*tk**2 
        delsq=delsq*(sigma8)**2/sigma8squn   
        pofk=2*np.pi**2*delsq
        return pofk
#--------------------------------
#
# Cosmic variance
#
#--------------------------------
    def Sigma8(self,k,Pk):
        '''
        Compute sigma8 with the simplest formula
        '''
        w = fourier.FT_sphere(8,k)
        integrand = Pk * w**2 * k**2
        sigma8 = np.sqrt( 1./ (2*np.pi**2) * integrate.simps(integrand,k))
        return sigma8

    def cv_sphere(self,k,Pk,R,nsteps=50):
        '''
        Compute the cosmic variance in a sphere of radius R using 3D integral
        '''
        kmin = np.min(k)
        #kmax = 2*np.pi/R
        kmax = np.max(k)

        Pk_int = interpolate.interp1d(k,Pk)
        w = lambda k: fourier.FT_sphere(R,k)
        norm = lambda k1,k2,k3: np.sqrt(k1**2+k2**2+k3**2)

        k3 = np.linspace(kmin,kmax,nsteps)
        k2 = np.linspace(kmin,kmax,nsteps)
        k1 = np.linspace(kmin,kmax,nsteps)
        ### 3D Integral ###
        Integrand1 = lambda k3,k2,k1: Pk_int(norm(k1,k2,k3)) * w(k1)**2 * w(k2)**2 * w(k3)**2 if ((norm(k1,k2,k3) < kmax) & (norm(k1,k2,k3) > kmin)) else 0        
        I3 = lambda k1,k2: integrate.simps(self.__Get_integrand(k3,Integrand1,k1,k2),k3)
        I2 = lambda k1: integrate.simps(self.__Get_integrand(k2,I3,k1),k2)
        I = integrate.simps(self.__Get_integrand(k1,I2),k1)*8 # *8 to account for negative wavenumber
        return np.sqrt(I/(8*np.pi**3))

    def cv_box(self,k,Pk,x1,x2,x3,nsteps=50):
        '''
        Compute the cosmic variance in a box of dimensions (x1,x2,x3)
        '''
        kmin = np.min(k)
        #kmax = 2*np.pi/R
        kmax = np.max(k)

        Pk_int = interpolate.interp1d(k,Pk)
        w1 = lambda k: fourier.FT_tophat(x1,k)
        w2 = lambda k: fourier.FT_tophat(x2,k)
        w3 = lambda k: fourier.FT_tophat(x3,k)
        norm = lambda k1,k2,k3: np.sqrt(k1**2+k2**2+k3**2)

        k1 = np.linspace(kmin,kmax,nsteps)
        k2 = np.linspace(kmin,kmax,nsteps)
        k3 = np.linspace(kmin,kmax,nsteps)
        
        ### 3D Integral ###
        Integrand1 = lambda k3,k2,k1: Pk_int(norm(k1,k2,k3)) * w1(k1)**2 * w2(k2)**2 * w3(k3)**2 if ((norm(k1,k2,k3) < kmax) & (norm(k1,k2,k3) > kmin)) else 0        
        I3 = lambda k1,k2: integrate.simps(self.__Get_integrand(k3,Integrand1,k1,k2),k3)
        I2 = lambda k1: integrate.simps(self.__Get_integrand(k2,I3,k1),k2)
        I = integrate.simps(self.__Get_integrand(k1,I2),k1)*8 # *8 to account for negative wavenumber
        return np.sqrt(I/(8*np.pi**3))
    
    def __Get_integrand(self,x,func,y=False,z=False):
        if z==False:
            if y==False:
                I = [func(ix) for ix in x]
            else:
                I = [func(ix,y) for ix in x]
        else:
            I = [func(ix,y,z) for ix in x]
        return I

#--------------------------------
#
# Plots
#
#--------------------------------
    def Plot_H(self,zmax,zmin=0,norm=True):
        '''
        Plot the Hubble parameter from zmin to zmax
        '''
        if zmax<=zmin:
            print('### ERRROR ### Plot_H : bad range')
            return
        step = (zmax-zmin)/float(1000)
        z = np.arange(zmin,zmax,step)
        H = []
        for iz in z:
            H.append(self.H(iz))
        if norm:
            p.plot(z,H/(1+z))
        else:
            p.plot(z,H)
        p.xlabel('redshift')
        if norm:
            p.ylabel('H(z)/(1+z) (km/s/(Mpc/h)')
        else:
            p.ylabel('H(z) (km/s/(Mpc/h))')
        p.show()
        return

    def Plot_distances(self,zmax,zmin=0, da=True, dl=True, chi=True, logscale=False):
        '''
        Plot comoving, angular and luminosity distances from zmin to zmax
        ''' 
        if zmax<=zmin:
            print('### ERRROR ### Plot_distances : bad range')
            return
        step = (zmax-zmin)/float(1000)
        z = np.arange(zmin,zmax,step)
        Chi,D_A,D_L = [],[],[]
        for iz in z:
            if chi:
                Chi.append(self.Comoving_distance(iz))
            if da:
                D_A.append(self.Angular_distance(iz))
            if dl:
                D_L.append(self.Luminosity_distance(iz))
        if chi:
            p.plot(z,Chi,label='Comoving')
        if da:
            p.plot(z,D_A,label='Angular')
        if dl:
            p.plot(z,D_L,label='Luminosity')
        p.legend(loc=2)
        if logscale:
            p.yscale('log')
        p.xlabel('redshift')
        p.ylabel('distance (Mpc/h)')
        p.show()
        return

    def Plot_angular_scale(self, l, zmax, zmin=0.01, logscale=True):
        '''
        Plot the angular scale as a function of redshift
        '''    
        if zmax<=zmin:
            print('### ERRROR ### Plot_angular scale : bad range')
            return
        if zmin == 0:
            zmin = 0.01
        step = (zmax-zmin)/float(1000)
        z = np.arange(zmin,zmax,step)
        D_A = []
        for iz in z:
            D_A.append(self.Angular_distance(iz))
        p.plot(z,l/(1+z)/np.array(D_A)*180/np.pi)
        if logscale:
            p.yscale('log')
        p.xlabel('redshift')
        p.ylabel('Angle (degrees)')
        p.show()
        return
    
    def Plot_pk(self,Pk,show=True):
        p.plot(Pk[0],Pk[1])
        p.xscale('log')
        p.yscale('log')
        if show:
            p.show()
        return
