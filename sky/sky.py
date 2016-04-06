import numpy as np

def Solid_angle(ra=[0,2*np.pi],dec=[-np.pi/2.,np.pi/2.],degrees=False):
    '''
    Return solid angle given ra and dec range
    '''
    if degrees == True:
        ra=np.array(ra)*np.pi/180
        dec=np.array(dec)*np.pi/180
    ang = np.abs(ra[1]-ra[0]) * ( np.sin(dec[1]) - np.sin(dec[0]) ) 
    if degrees == True:
        return ang*180**2/np.pi**2
    else:
        return ang

