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
        self.distance_unit = 'Mpc'
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
        h = self.H(z)/100.
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
        return {'Omega_m_h2':'none', 'Omega_b_h2':'none', 'Omega_l':'none', 'Omega_r':'none', 'h':'none', 'c':'km/s', 'Theta27':'2.7K'}

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
        Return current distance unit
        '''
        return self.distance_unit

    def Set_distance_unit(self,unit):
        '''
        Set distance unit. Choices are 'Mpc' or 'Mpc/h'
        '''
        unit_list = ['Mpc','Mpc/h']
        if unit not in unit_list:
            print('### ERROR ### Set_distance_unit :',unit,'not available. Choices are',unit_list)
        else:
            self.distance_unit = unit
        return

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
        if self.distance_unit == 'Mpc':
            return self.h*hubble
        if self.distance_unit == 'Mpc/h':
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
            z_0,z_1 = z[0],z[1]
        I = integrate.quad(lambda x: 1./self.H(x),z_0,z_1)
        if I[1]/I[0]>0.01:
            print('### WARNING ### Comoving_distance : estimate has error of',I[1]/I[0]*100,'%' )
        Chi = I[0]*self.c
        return Chi

    def Tranverse_comoving_distance(self,z):
        '''
        Compute the tranverse comoving distance (chi, sin(chi) or sinh(chi) depending on Omega_k)
        '''
        Ok = self.Omega_k()
        d_H = self.c/(100*self.h)
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
# Volumes
#
#--------------------------------
    def Comoving_volume(self,z,ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
        '''
        Compute the comoving volume given a redshift range and angles on the sky
        '''
        Chi = self.Comoving_distance(z[1]) - self.Comoving_distance(z[0])
        ang = sky.Solid_angle(ra,dec,degrees)
        if degrees == True:
            ang*=np.pi**2/180**2
        return ang*1./3*Chi**3

    def Comoving_surface(self,z,ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
        '''
        Compute the comoving surface at given redshift
        '''
        ang = sky.Solid_angle(ra,dec,degrees)
        if degrees == True:
            ang*=np.pi**2/180**2
        return ( (1+z)*self.Angular_distance(z) )**2 * ang


    def Comoving_volume2(self,z,ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
        '''
        Compute the comoving volume given a redshift range and angles on the sky (similar as Comoving_volume)
        '''
        I = integrate.quad(lambda x: self.Comoving_surface(x,ra,dec,degrees)*misc.derivative(self.Comoving_distance,x,dx=1e-6),z[0],z[1])
        return I[0]

#--------------------------------
#
# Eisenstein & Hu
#
#--------------------------------
    def s(self):
        '''
        Sound_horizon - Fitting formula from Eisenstein & Hu 1998 that says 'approximate the sound horizon to ~2% over the range Omega_b_h2 > 0.0125 and 0.025 < Omega_m_h2 < 0.5'.
        Computed from the values of Omega_b_h2 and Omega_c_h2 of the Universe class.
        '''
        s = 44.5*np.log(9.83/self.Omega_m_h2) / np.sqrt(1+10*(self.Omega_b_h2)**(3./4))
        return s
    
    def T(self,k):
        '''
        Eisenstein & Hu 1998
        transfer function
        '''
        T = self.Omega_b_h2/self.Omega_m_h2*self.Tb(k) + (self.Omega_m_h2 - self.Omega_b_h2)/self.Omega_m_h2*self.Tc(k)
        return T

    def Tc(self,k):
        '''
        Eisenstein & Hu 1998
        CDM transfer function
        '''
        Tc = self.f(k)*self.Ttild(k,1,self.betac()) + (1-self.f(k))*self.Ttild(k,self.alphac(),self.betac())
        return Tc

    def alphac(self):
        '''
        Eisenstein & Hu 1998
        '''
        a1 = (46.9*self.Omega_m_h2)**0.670 * (1+(32.1*self.Omega_m_h2)**(-0.532))
        a2 = (12.0*self.Omega_m_h2)**0.424 * (1+(45.0*self.Omega_m_h2)**(-0.582))
        ac = a1**(-self.Omega_b_h2/self.Omega_m_h2)*a2**(-(self.Omega_b_h2/self.Omega_m_h2)**3)
        return ac

    def betac(self):
        '''
        Eisenstein & Hu 1998
        '''
        b1 = 0.944*(1+(458*self.Omega_m_h2)**(-0.708))**(-1)
        b2 = (0.395*self.Omega_m_h2)**(-0.0266)
        ibc = 1 + b1*( ((self.Omega_m_h2 - self.Omega_b_h2)/self.Omega_m_h2)**b2 - 1)
        return 1./ibc

    def f(self,k):
        '''
        Eisenstein & Hu 1998
        '''
        f = 1./(1 + (k*self.s()/5.4)**4)
        return f

    def Ttild(self,k,ac,bc):
        '''
        Eisenstein & Hu 1998
        '''
        C = 14.2/ac + 386./(1 + 69.9*self.q(k)**1.08)
        Ttild = np.log(np.e + 1.8*bc*self.q(k))/( np.log(np.e + 1.8*bc*self.q(k)) + C*self.q(k)**2)
        return Ttild

    def q(self,k):
        '''
        Eisenstein & Hu 1998
        '''
        q = k/ (13.41*self.keq())
        return q

    def keq(self):
        '''
        Eisenstein & Hu 1998
        '''
        keqp = (2*self.Omega_m_h2*100**2*self.zeq())**(1./2)
        keq = 7.46*10**(-2)*self.Omega_m_h2*self.Theta27**(-2) 
        return keq

    def zeq(self):
        '''
        Eisenstein & Hu 1998
        '''
        zeq = 2.50*10**4*self.Omega_m_h2 * self.Theta27**(-4)
        return zeq 
    
    def Tb(self,k):
        '''
        Eisenstein & Hu 1998
        Baryonic transfer function
        '''
        Tb = ( self.Ttild(k,1,1)/(1+(k*self.s()/5.2)**2) + self.alphab()/( 1 + (self.betab()/(k*self.s()))**3 ) * np.e**( -(k/self.ksilk())**1.4 ) )*np.sin(k*self.stild(k))/k/self.stild(k)
        return Tb

    def alphab(self):
        '''
        Eisenstein & Hu 1998
        '''
        alphab = 2.07*self.keq()*self.s()*(1+self.R(self.zd()))**(-3./4)*self.G((1+self.zeq())/(1+self.zd()))
        return alphab

    def G(self,y):
        '''
        Eisenstein & Hu 1998
        '''
        G = y*(-6*np.sqrt(1+y) + (2 + 3*y)*np.log( (np.sqrt(1+y) + 1) / (np.sqrt(1+y) - 1) ) )
        return G

    def betab(self):
        '''
        Eisenstein & Hu 1998
        '''
        bb = 0.5 + self.Omega_b_h2/self.Omega_m_h2 + (3 - 2*self.Omega_b_h2/self.Omega_m_h2)*np.sqrt((17.2*self.Omega_m_h2)**2 + 1)
        return bb

    def zd(self):
        '''
        Eisenstein & Hu 1998
        '''
        b1 = 0.313*(self.Omega_m_h2)**(-0.419)*(1 + 0.607*(self.Omega_m_h2)**0.674)
        b2 = 0.238*(self.Omega_m_h2)**0.223
        zd = 1291 * self.Omega_m_h2**0.251 / (1 + 0.659*self.Omega_m_h2**0.828) * (1 + b1*self.Omega_b_h2**b2)
        return zd

    def R(self,z):
        '''
        Eisenstein & Hu 1998
        '''
        R = 31.5*self.Omega_b_h2*self.Theta27**(-4)*(z/10**3)**(-1)
        return R

    def ksilk(self):
        '''
        Eisenstein & Hu 1998
        '''
        ksilk = 1.6 * (self.Omega_b_h2)**0.52 * (self.Omega_m_h2)**0.73 * (1 + (10.4*self.Omega_m_h2)**(-0.95))
        return ksilk

    def stild(self,k):
        '''
        Eisenstein & Hu 1998
        '''
        stild = self.s() / (1 + (self.betanode()/k/self.s())**3)**(1./3)
        return stild

    def betanode(self):
        '''
        Eisenstein & Hu 1998
        '''
        betanode = 8.41*self.Omega_m_h2**0.435
        return betanode

#--------------------------------
#
# Pk
#
#--------------------------------
    def Sigma8(self,k,Pk):
        w = 3 / (k*8)**3 * (np.sin(8*k) - 8*k*np.cos(8*k))
        integrand = Pk * w**2 * k**2
        sigma8 = np.sqrt( 1./ (2*np.pi**2) * integrate.simps(integrand,k))
        return sigma8

    def Pk_camb(self,z=0):
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
        print('### INFO ### Pk_camb : Invoking camb')
        os.system(os.path.join('./camb')+' '+os.path.join(self.camb_path,inifile))
        os.system('cd '+self.camb_path)
        camb_results = ascii.read(os.path.join(self.camb_path,outfile))
        k = camb_results['col1'].data
        Pk = camb_results['col2'].data
        '''
        if self.distance_unit == 'Mpc/h':
            k/=params['h']
            Pk*=params['h']**3
            '''
        return (k,Pk)

    def Pk_growth(self,z=0,sig_v=4.48,damping=True):
        '''
        Compute Pk at redshift z (default z=0) using Pk from Camb at z=0 with cosmological parameters of the class plus linear growth factor and damping.
        
        Input:
        - z (=0)            : Redshift at which Pk is evaluated
        - damping (=True)   : Whether to include streaming damping
        - sig_v (=4.48)     : Value of the streaming scale. Default is 4.47 Mpc = 3 Mpc/h for h = 0.67 which is the value of Anderson et al. 2014

        Output:
        - [k,Pk]            : In Mpc^3
        '''
        Pk0 = self.Pk_camb()
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
        I0 = integrate.quad(lambda x: (1+x)/(self.H(x)/(self.h*100))**3,0,3000)
        I = integrate.quad(lambda x: (1+x)/(self.H(x)/(self.h*100))**3,z,3000)
        if I[1]/I[0]>0.01:
            print('### WARNING ### Linear_growth : estimate has error of',I[1]/I[0]*100,'%' )
        D0 = 5./2*self.Omega_m_h2/self.h**2 * I0[0]
        D = 5./2*self.Omega_m_h2/self.h**2 * self.H(z)/(self.h*100) * I[0]
        return D/D0
                  
    def Pk_th(self,k,z=0):
        '''
        Compute the theoretical power spectrum using the Eisenstein & Hu transfer function 
        '''
        Pk = k**self.n_s * self.T(k) * self.Linear_growth(z)
        return Pk

    def Sampler(self):
        '''
        Internal - Sampler for Cosmic_variance_MC mcint. Not to be called by user.
        '''
        while True:
            k1 = random.uniform(0.,np.sqrt((self.kmax**2)/3.))
            k2 = random.uniform(0.,np.sqrt((self.kmax**2)/3.))
            k3 = random.uniform(0.,np.sqrt((self.kmax**2)/3.))
            yield (k1,k2,k3)

    def Cosmic_variance_MC2(self,k,Pk,z,width,height,degrees=True,nsteps=50):
        '''
        Compute the cosmic variance using a Monte Carlo approch
        '''
        zmean = (z[1]+z[0])/2.

        Pk_int = interpolate.interp1d(k,Pk*self.Linear_growth(zmean))

        norm = lambda k1,k2,k3: np.sqrt(k1**2+k2**2+k3**2)

        x1 = (1+zmean)*self.Angular_distance(zmean)*width
        x2 = (1+zmean)*self.Angular_distance(zmean)*height
        x3 = self.Comoving_distance(z[1]) - self.Comoving_distance(z[0]) 

        #DEBUG
        if debug==True:
            x1 = 8
            x2 = 8
            x3 = 8
            kmax1 = 2*np.pi/(x1)
            kmax2 = 2*np.pi/(x2)
            kmax3 = 2*np.pi/(x3)
        else:
            kmax1 = 2*np.pi/(x1/2.)
            kmax2 = 2*np.pi/(x2/2.)
            kmax3 = 2*np.pi/(x3/2.)
        
#        kmax = norm(2*np.pi/(3*x1),2*np.pi/(3*x2),2*np.pi/(3*x3))
        '''
        w1 = lambda k: np.sin(np.pi*x1*k) / (np.pi*k) 
        w2 = lambda k: np.sin(np.pi*x2*k) / (np.pi*k) 
        w3 = lambda k: np.sin(np.pi*x3*k) / (np.pi*k) 
        '''
        w1 = lambda k1: np.sin(x1*k1/2.) / (k1*x1/2.) 
        w2 = lambda k2: np.sin(x2*k2/2.) / (k2*x2/2.) 
        w3 = lambda k3: np.sin(x3*k3/2.) / (k3*x3/2.) 

        #w = lambda k1,k2,k3: w1(k1) * w2(k2) * w3(k3)

#        w = lambda k1,k2,k3: np.abs(w1(k1)) * np.abs(w2(k2)) * np.abs(w3(k3))
#        Integrand = lambda k1,k2,k3: Pk_int( norm(k1,k2,k3) ) * w(k1,k2,k3)**2

        kmin = k[0]
#        kmax_1d = np.sqrt((kmax**2)/3.)


        ### 1D Integral ###
        '''
        w = lambda k1: 3 / (k1*x1)**3 * (np.sin(x1*k1) - x1*k1*np.cos(x1*k1))
        Integrand = lambda k1: k1**2 * Pk_int(k1) * w(k1)**2
        I = self.Integral_trapez(Integrand,kmin,kmax1,nsteps=nsteps,dim=1)
        I*= 1./ (2*np.pi**2)
        '''
        '''
        Test = lambda k1,k2,k3: 1 if (k1<1) & (k2<1) & (k3<1) else 0
        I = self.Integral_trapez(Test,0,3,0,3,0,3,nsteps=nsteps,dim=3)  
        '''
        ### 3D Integral ###
        w = lambda k1: 3 / (k1*x1)**3 * (np.sin(x1*k1) - x1*k1*np.cos(x1*k1))
        Integrand = lambda k1,k2,k3: Pk_int(norm(k1,k2,k3)) * w(norm(k1,k2,k3))**2
        I = self.Integral_trapez(Integrand,kmin,kmax1,kmin,kmax2,kmin,kmax3,nsteps=nsteps,dim=3)
        I*= 1./ (2*np.pi)**3
        return np.sqrt(I)
    
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
    
    def Cosmic_variance_MC3(self,k,Pk,z,width,height,degrees=True,nsteps=50,window='tophat',debug=False):
        '''
        Compute the cosmic variance using scipy Simpson integration
        '''
        if degrees==True:
            width*=np.pi/180.
            height*=np.pi/180.

        zmean = (z[1]+z[0])/2.
        Pk_int = interpolate.interp1d(k,Pk*self.Linear_growth(zmean))
#        Pk_int = lambda k: self.Pk_Jeff(k)*self.Linear_growth(zmean)
        norm = lambda k1,k2,k3: np.sqrt(k1**2+k2**2+k3**2)
        x1 = (1+zmean)*self.Angular_distance(zmean)*width
        x2 = (1+zmean)*self.Angular_distance(zmean)*height
        x3 = self.Comoving_distance(z[1]) - self.Comoving_distance(z[0]) 

        #DEBUG
        if debug==True:
            x1 = 8
            x2 = 8
            x3 = 8
            kmax1 = 2*np.pi/(x1)
            kmax2 = 2*np.pi/(x2)
            kmax3 = 2*np.pi/(x3)
        else:
            '''
            kmax1 = 2*np.pi/(x1/2.)
            kmax2 = 2*np.pi/(x2/2.)
            kmax3 = 2*np.pi/(x3/2.)
            '''

            kmax1 = 2*np.pi/(x1/20.)
            kmax2 = 2*np.pi/(x2/20.)
            kmax3 = 2*np.pi/(x3/20.)

        w1 = lambda k1: np.sin(x1*k1/2.) / (k1*x1/2.) 
        w2 = lambda k2: np.sin(x2*k2/2.) / (k2*x2/2.) 
        w3 = lambda k3: np.sin(x3*k3/2.) / (k3*x3/2.) 
        '''
        w1 = lambda k1: np.sin(np.pi*x1*k1) / (np.pi*k1*x1) 
        w2 = lambda k2: np.sin(np.pi*x2*k2) / (np.pi*k2*x2) 
        w3 = lambda k3: np.sin(np.pi*x3*k3) / (np.pi*k3*x3) 
        '''
        kmin = k[0]

        k3 = np.linspace(kmin,kmax3,nsteps)
        k2 = np.linspace(kmin,kmax2,nsteps)
        k1 = np.linspace(kmin,kmax1,nsteps)

        ### 3D Integral ###
        if window=='sphere':
            w = lambda k1,k2,k3: 3 / (norm(k1,k2,k3)*x1)**3 * (np.sin(x1*norm(k1,k2,k3)) - x1*norm(k1,k2,k3)*np.cos(x1*norm(k1,k2,k3)))
#        w = lambda k1,k2,k3: w1(k1) * w2(k2) * w3(k3) if 1./norm(k1,k2,k3)< 8 else 0
        elif window=='tophat':
            w = lambda k1,k2,k3: w1(k1) * w2(k2) * w3(k3)
        else:
            print("### Error ### Cosmic_variance_MC3: wrong window choices, possibilities are tophat (defaukt) or sphere")
        Integrand1 = lambda k3,k2,k1: Pk_int(norm(k1,k2,k3)) * w(k1,k2,k3)**2        

        I3 = lambda k1,k2: integrate.simps(self.Get_integrand(k3,Integrand1,k1,k2),k3)
        I2 = lambda k1: integrate.simps(self.Get_integrand(k2,I3,k1),k2)
        I = integrate.simps(self.Get_integrand(k1,I2),k1)
#        I3 = lambda k2,k1: integrate.quad(Integrand,kmin,kmax3,args=(k2,k1))[0]
#        I2 = lambda k1: integrate.quad(I3,kmin,kmax2,args=(k1))[0]
#        I = integrate.quad(I2,kmin,kmax1)[0]
 
        I*= 1./ (np.pi)**3
        return np.sqrt(I)

    def Get_integrand(self,x,func,y=False,z=False):
        if z==False:
            if y==False:
                I = [func(ix) for ix in x]
            else:
                I = [func(ix,y) for ix in x]
        else:
            I = [func(ix,y,z) for ix in x]
        return I

    def Integral_trapez(self,func,x_min,x_max,y_min=0,y_max=0,z_min=0,z_max=0,nsteps=100,dim=3):
        '''
        Compute the 1d, 2d or 3d integral of func using trapezoidal integration

        Inputs:
        - func         : a function or method
        - x_min (float): inferior limit on x
        - x_max (float): superior limit on x
        - y_min (float): inferior limit on y
        - y_max (float): inferior limit on y 
        - z_min (float): inferior limit on z
        - z_max (float): inferior limit on z
        - nsteps (int) : number of steps
        - dim    (int) : dimension of the integral

        Outputs:
        - Value of the integral
        '''
        dx = (x_max - x_min)/float(nsteps)
        dy = (y_max - y_min)/float(nsteps)
        dz = (z_max - z_min)/float(nsteps)

        ind = np.arange(nsteps)
        x = ind*dx + x_min
        y = ind*dy + y_min
        z = ind*dz + z_min
        
        
        if (dim==3):
#       I = dx*dy*dz/8. * np.array([ func(x[i],y[j],z[k]) + func(x[i],y[j],z[k+1]) + func(x[i],y[j+1],z[k]) + func(x[i],y[j+1],z[k+1]) + func(x[i+1],y[j],z[k]) + func(x[i+1],y[j],z[k+1]) + func(x[i+1],y[j+1],z[k]) + func(x[i+1],y[j+1],z[k+1]) for i,j,k in itertools.product(ind[:-1],ind[:-1],ind[:-1])])
            iter2 = itertools.product(ind[1:-1],ind[1:-1])
            iter3 = itertools.product(ind[1:-1],ind[1:-1],ind[1:-1])
            I1 = func(x[0],y[0],z[0]) + func(x[0],y[0],z[nsteps-1]) + func(x[0],y[nsteps-1],z[0]) + func(x[0],y[nsteps-1],z[nsteps-1]) + func(x[nsteps-1],y[0],z[0]) + func(x[nsteps-1],y[0],z[nsteps-1]) + func(x[nsteps-1],y[nsteps-1],z[0]) + func(x[nsteps-1],y[nsteps-1],z[nsteps-1])
            I2_z = np.sum([func(x[0],y[0],z[i]) + func(x[0],y[nsteps-1],z[i]) + func(x[nsteps-1],y[0],z[i]) + func(x[nsteps-1],y[nsteps-1],z[i]) for i in ind[1:-1]])
            I2_y = np.sum([func(x[0],y[i],z[0]) + func(x[0],y[i],z[nsteps-1]) + func(x[nsteps-1],y[i],z[0]) + func(x[nsteps-1],y[i],z[nsteps-1]) for i in ind[1:-1]])
            I2_x = np.sum([func(x[i],y[0],z[0]) + func(x[i],y[0],z[nsteps-1]) + func(x[i],y[nsteps-1],z[0]) + func(x[i],y[nsteps-1],z[nsteps-1]) for i in ind[1:-1]])
            I4_yz = np.sum([func(x[0],y[i],z[j]) +func(x[nsteps-1],y[i],z[j]) for i,j in iter2])
            I4_xz = np.sum([func(x[i],y[0],z[j]) +func(x[i],y[nsteps-1],z[j]) for i,j in iter2])
            I4_xy = np.sum([func(x[i],y[j],z[0]) +func(x[i],y[j],z[nsteps-1]) for i,j in iter2])
            I8 = np.sum([func(x[i],y[j],z[k]) for i,j,k in iter3])
            I = dx*dy*dz/8. * (I1 + 2*(I2_x+I2_y+I2_z) + 4*(I4_xy+I4_xz+I4_yz) + 8*I8)
        elif (dim==2):
            I = dx*dy/4. * np.array([func(x[i],y[j]) + func(x[i],y[j+1]) + func(x[i+1],y[j]) + func(x[i+1],y[j+1]) for i,j in itertools.product(ind[:-1],ind[:-1])])
        elif (dim==1):
            I = dx/2. * np.array([func(x[i]) + func(x[i+1]) for i in ind[:-1]])
        else:
            print('### ERROR ### Integral_trapez: wrong dimension, please use dim = 1,2 or 3')
            return
        return np.sum(I)


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
            p.ylabel('H(z)/(1+z) (km/s/Mpc)')
        else:
            p.ylabel('H(z) (km/s/Mpc)')
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
        p.ylabel('distance ('+self.distance_unit+')')
        p.show()
        return

    def Plot_angular_scale(self, l, zmax, zmin=0, logscale=True):
        '''
        Plot the angular scale as a function of redshift
        '''    
        if zmax<=zmin:
            print('### ERRROR ### Plot_angular scale : bad range')
            return
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




















### OBSOLETE ###
    def Box(self,height,width,z,box_size=10,nbins=200,degrees=True):
        '''
        Create a 3d array (a box) that is *box_size* times bigger than the volume that has ones inside the volume and 0 outside
        
        Inputs:
        - height (float)   : angle  
        - width (float)    : angle  
        - z (float array len 2)  : redshift limits
        - box_size (int)   : number of times the box is bigger than the volume of interest in any direction
        - nbins (int)      : number of bins in the smallest direction

        Returns:
        - step_size (float) : the comoving distance between to pixels  
        - box (float array of dim3) : the actual box
        '''
        if degrees:
            height*=np.pi/180
            width*=np.pi/180
        # define x,y and z length of the volume
        mean_z = (z[1]+z[0])/2
        xv = (1+mean_z)*self.Angular_distance(mean_z)*height
        yv = (1+mean_z)*self.Angular_distance(mean_z)*width
        zv = self.Comoving_distance(z[1]) - self.Comoving_distance(z[0])
        # define x,y and z length of the box
        x = xv*box_size
        y = yv*box_size
        z = zv*box_size
        # Compute number of bins per axis
        step_size = min([x,y,z])/nbins
        if step_size < 0.5:
            step_size = 0.5
        box_x = int(x/step_size)
        box_y = int(y/step_size)
        box_z = int(z/step_size)
        # Create box and set center
        box = np.zeros((box_z,box_x,box_y))
        x_center = box_x/2
        y_center = box_y/2
        # Compute max distance to the center in bins
        xv_bins = xv/step_size/2.
        yv_bins = yv/step_size/2.
        # Fill ones
        for iz in np.arange(box_z):
            if iz*step_size > zv:
                continue
            for ix in np.arange(box_x):
                if np.abs(ix-x_center) > xv_bins:
                    continue
                for iy in np.arange(box_y):
                    if np.abs(iy-y_center) > yv_bins:
                        continue
                    box[iz,ix,iy] = 1
        return step_size, box

    def Box_fft(self,box,step_size):
        '''
        Do the FFT of a box
        '''
        ftbox = fft.fftn(box)
        wave1 = fft.fftfreq(box.shape[0],d=step_size)
        wave2 = fft.fftfreq(box.shape[1],d=step_size)
        wave3 = fft.fftfreq(box.shape[2],d=step_size)
        # Keep only real part
        wave1 = wave1[:box.shape[0]/2]
        wave2 = wave2[:box.shape[1]/2]
        wave3 = wave3[:box.shape[2]/2]
        ftbox = ftbox[:box.shape[0]/2,:box.shape[1]/2,:box.shape[2]/2]
        self.window_func = np.abs(ftbox)
        self.window_wave = [wave1,wave2,wave3]
        return [wave1,wave2,wave3],np.abs(ftbox)

    def Cosmic_variance_MC(self,Pk,z,width,height,degrees=True,nmc=1e5):
        '''
        Compute the cosmic variance using a Monte Carlo approch
        '''
        self.Pk = Pk
        print('Computing window function')
        scale,box = self.Box(height,width,z,degrees=degrees)
        print('Computing fft of window function')
        wave,fbox = self.Box_fft(box,scale)
        self.window_func = fbox
        self.window_wave = wave
        print('Computing integral of Pk')
        domaine_size = wave[0][-1]*wave[1][-1]*wave[2][-1]
        random.seed(1)
        results, error = mcint.integrate(self.Integrand_MC, self.Sampler(), measure=domaine_size, n=nmc)
        results/=8*np.pi**3
        print('Done')
        return (results,error)



    def Integrand(self,k1,k2,k3,Pk,wave,fbox):
        '''
        Integrand for cosmic variance computation, not to be called directly
        '''
        norm_k = np.sqrt(k1**2+k2**2+k3**2)
        Int = interpolate.interp1d(Pk[0], Pk[1])(norm_k) 

        Int*=(interpolate.interpn(wave,fbox,[k1,k2,k3])[0])**2
        return Int

    def Cosmic_variance(self,Pk,z,width,height,degrees=True):
        '''
        Compute the cosmic variance
        '''
        print('Computing window function')
        scale,box = self.Box(height,width,z,degrees=degrees)
        print('Computing fft of window function')
        wave,fbox = self.Box_fft(box,scale)
        print('Computing integral of Pk')
        CV = 1./8/np.pi**3*integrate.tplquad(self.Integrand,wave[0][0],wave[0][-1],lambda x:wave[1][0],lambda x:wave[1][-1],lambda x,y:wave[2][0],lambda x,y:wave[2][-1],args=(Pk,wave,fbox))
        print('Done')
        return CV

    def Integrand_MC(self,k):
        '''
        Internal - Integrand for cosmic variance computation, not to be called by user.
        '''
        k1 = k[0]
        k2 = k[1]
        k3 = k[2]
        norm_k = np.sqrt(k1**2+k2**2+k3**2)
        Int = interpolate.interp1d(self.Pk[0], self.Pk[1])(norm_k) 
        Int*=(interpolate.interpn(self.window_wave,self.window_func,[k1,k2,k3])[0])**2
        return Int

