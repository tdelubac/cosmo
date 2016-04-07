#--------------------------------
#
# Eisenstein & Hu
#
#--------------------------------
import numpy as np

def Pk(universe,k,z=0):
    '''
    Compute the theoretical power spectrum using the Eisenstein & Hu transfer function 
    Assume flatness through computation of delta_h (Eisenstein & Hu 1998 eq: A3)
    k : array of values at which to compute P(k).
    optional:
    z : redshift at which to compute P(k). Default 0. 
    '''
    Pk = 2*(np.pi)**2 * (__delta_h(universe))**2 * (universe.c / (100*universe.h))**(3+universe.n_s) * k**universe.n_s * (__T(universe,k))**2 * universe.Linear_growth(z)
    return Pk


def s(universe):
    '''
    Sound_horizon - Fitting formula from Eisenstein & Hu 1998 that says 'approximate the sound horizon to ~2% over the range Omega_b_h2 > 0.0125 and 0.025 < Omega_m_h2 < 0.5'.
    Computed from the values of Omega_b_h2 and Omega_c_h2 of the Universe class.
    '''
    s = 44.5*np.log(9.83/universe.Omega_m_h2) / np.sqrt(1+10*(universe.Omega_b_h2)**(3./4))
    return s

def __T(universe,k):
    '''
    Eisenstein & Hu 1998
    transfer function
    '''
    T = universe.Omega_b_h2/universe.Omega_m_h2*__Tb(universe,k) + (universe.Omega_m_h2 - universe.Omega_b_h2)/universe.Omega_m_h2*__Tc(universe,k)
    return T

def __Tc(universe,k):
    '''
    Eisenstein & Hu 1998
    CDM transfer function
    '''
    Tc = __f(universe,k)*__Ttild(universe,k,1,__betac(universe)) + (1-__f(universe,k))*__Ttild(universe,k,__alphac(universe),__betac(universe))
    return Tc

def __alphac(universe):
    '''
    Eisenstein & Hu 1998
    '''
    a1 = (46.9*universe.Omega_m_h2)**0.670 * (1+(32.1*universe.Omega_m_h2)**(-0.532))
    a2 = (12.0*universe.Omega_m_h2)**0.424 * (1+(45.0*universe.Omega_m_h2)**(-0.582))
    ac = a1**(-universe.Omega_b_h2/universe.Omega_m_h2)*a2**(-(universe.Omega_b_h2/universe.Omega_m_h2)**3)
    return ac

def __betac(universe):
    '''
    Eisenstein & Hu 1998
    '''
    b1 = 0.944*(1+(458*universe.Omega_m_h2)**(-0.708))**(-1)
    b2 = (0.395*universe.Omega_m_h2)**(-0.0266)
    ibc = 1 + b1*( ((universe.Omega_m_h2 - universe.Omega_b_h2)/universe.Omega_m_h2)**b2 - 1)
    return 1./ibc

def __f(universe,k):
    '''
    Eisenstein & Hu 1998
    '''
    f = 1./(1 + (k*s(universe)/5.4)**4)
    return f

def __Ttild(universe,k,ac,bc):
    '''
    Eisenstein & Hu 1998
    '''
    C = 14.2/ac + 386./(1 + 69.9*__q(universe,k)**1.08)
    Ttild = np.log(np.e + 1.8*bc*__q(universe,k))/( np.log(np.e + 1.8*bc*__q(universe,k)) + C*__q(universe,k)**2)
    return Ttild

def __q(universe,k):
    '''
    Eisenstein & Hu 1998
    '''
    q = k/ (13.41*__keq(universe))
    return q

def __keq(universe):
    '''
    Eisenstein & Hu 1998
    '''
    keqp = (2*universe.Omega_m_h2*100**2*__zeq(universe))**(1./2)
    keq = 7.46*10**(-2)*universe.Omega_m_h2*universe.Theta27**(-2) 
    return keq

def __zeq(universe):
    '''
    Eisenstein & Hu 1998
    '''
    zeq = 2.50*10**4*universe.Omega_m_h2 * universe.Theta27**(-4)
    return zeq 

def __Tb(universe,k):
    '''
    Eisenstein & Hu 1998
    Baryonic transfer function
    '''
    Tb = ( __Ttild(universe,k,1,1)/(1+(k*s(universe)/5.2)**2) + __alphab(universe)/( 1 + (__betab(universe)/(k*s(universe)))**3 ) * np.e**( -(k/__ksilk(universe))**1.4 ) )*np.sin(k*__stild(universe,k))/k/__stild(universe,k)
    return Tb

def __alphab(universe):
    '''
    Eisenstein & Hu 1998
    '''
    alphab = 2.07*__keq(universe)*s(universe)*(1+__R(universe,__zd(universe)))**(-3./4)*__G((1+__zeq(universe))/(1+__zd(universe)))
    return alphab

def __G(y):
    '''
    Eisenstein & Hu 1998
    '''
    G = y*(-6*np.sqrt(1+y) + (2 + 3*y)*np.log( (np.sqrt(1+y) + 1) / (np.sqrt(1+y) - 1) ) )
    return G

def __betab(universe):
    '''
    Eisenstein & Hu 1998
    '''
    bb = 0.5 + universe.Omega_b_h2/universe.Omega_m_h2 + (3 - 2*universe.Omega_b_h2/universe.Omega_m_h2)*np.sqrt((17.2*universe.Omega_m_h2)**2 + 1)
    return bb

def __zd(universe):
    '''
    Eisenstein & Hu 1998
    '''
    b1 = 0.313*(universe.Omega_m_h2)**(-0.419)*(1 + 0.607*(universe.Omega_m_h2)**0.674)
    b2 = 0.238*(universe.Omega_m_h2)**0.223
    zd = 1291 * universe.Omega_m_h2**0.251 / (1 + 0.659*universe.Omega_m_h2**0.828) * (1 + b1*universe.Omega_b_h2**b2)
    return zd

def __R(universe,z):
    '''
    Eisenstein & Hu 1998
    '''
    R = 31.5*universe.Omega_b_h2*universe.Theta27**(-4)*(z/10**3)**(-1)
    return R

def __ksilk(universe):
    '''
    Eisenstein & Hu 1998
    '''
    ksilk = 1.6 * (universe.Omega_b_h2)**0.52 * (universe.Omega_m_h2)**0.73 * (1 + (10.4*universe.Omega_m_h2)**(-0.95))
    return ksilk

def __stild(universe,k):
    '''
    Eisenstein & Hu 1998
    '''
    stild = s(universe) / (1 + (__betanode(universe)/k/s(universe))**3)**(1./3)
    return stild

def __betanode(universe):
    '''
    Eisenstein & Hu 1998
    '''
    betanode = 8.41*universe.Omega_m_h2**0.435
    return betanode

def __delta_h(universe):
    '''
    Eisenstein & Hu 1998 (eq : A3)
    Assumes flatness
    '''
    exp = -0.785 - 0.05 * np.log(universe.Omega_m_h2 / universe.h**2) 
    delta_h = 1.94*10**(-5) * (universe.Omega_m_h2 / universe.h**2)**exp * np.e**(-0.95*(universe.n_s-1) - 0.169*(universe.n_s-1)**2)
    return delta_h

