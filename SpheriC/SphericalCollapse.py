'''module for computing the spherical collapse model under arbitrary cosmology
   The SCSolver can predict the turnaround overdensity, Virial density, and cosmic age at any given scalefactor.
   
   Example:
   
        from SphericalCollapse import SCSolver
        OmegaM=0.268
        a=0.8 #scalefactor
        s=SCSolver(OmegaM) #assuming flat universe if OmegaL not given.
        s.TurnaroundOverdensity(a)
        s.VirialDelta(a)
        s.Age(a)

Author: 
  Jiaxin Han   
  jiaxin.han@sjtu.edu.cn
  22/09/2021
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

def ArateSquare(a, w, kappa):
    '''(da/dtau)^2, where a is the scalefactor, tau=sqrt(OmegaM0)*H0*t
       w: OmegaL0/OmegaM0
       kappa: the energy (or curvature) parameter
    '''
    return 1./a+w*a*a-kappa 

def ArateInv(a, w, kappa):
    '''dtau/da, where a is the scalefactor, tau=sqrt(OmegaM0)*H0*t
       w: OmegaL0/OmegaM0
       kappa: the energy (or curvature) parameter
    '''
    if a==0.:
        return 0.
    return 1./np.sqrt(ArateSquare(a,w,kappa)) 

def TimeIntegral(a, w, kappa):
    '''tau=sqrt(OmegaM0)*H0*t, at scalefactor a'''
    return integrate.quad(ArateInv, 0, a, (w, kappa))[0]


def TimeIntegralTa(ap_ta, w):
    '''tau=sqrt(OmegaM0)*H0*t, at turnaround scalefactor ap_ta'''
    kappa=ArateSquare(ap_ta, w, 0)
    return TimeIntegral(ap_ta, w, kappa)

def TimeIntegralAp(ap, ap_ta, w, side='left'):
    '''tau=sqrt(OmegaM0)*H0*t, integrated from 0 to ap, for a perturbation which turnaround at ap_ta
    side= 'left' or 'right', whether ap is before or after ap_ta.
    '''
    kappa=ArateSquare(ap_ta, w, 0)
    if side=='left':
        return TimeIntegral(ap, w, kappa)
    elif side=='right':
        return 2*TimeIntegral(ap_ta, w, kappa)-TimeIntegral(ap, w, kappa)
    else:
        raise TypeError("side must be left or right")

class SCSolver(object):
    def __init__(self, OmegaM0, OmegaL0=None, a_max=1.):
        '''cosmological parameters at a=1 (z=0)
        If OmegaL0 is not given, then set to 1-OmegaM0.
        a_max: maximum scalefactor at which the model will be applied to. This is only required for Lambda=0 universe.
        '''
        if OmegaL0 is None:
            OmegaL0=1-OmegaM0
        self.OmegaM0=OmegaM0
        self.OmegaL0=OmegaL0
        self.OmegaK0=1-OmegaL0-OmegaM0
        self.w=OmegaL0/OmegaM0
        self.wk=-self.OmegaK0/OmegaM0
        self.a_max=a_max
        if self.w==0:
            self.ap_max=a_max #as TimeIntegralTa(a)>TimeIntegral(a) for bound regions, setting ap_max to a_max is sufficient for building the lookup table
        else:
            self.ap_max=(0.5/self.w)**(1./3)
        
        #build lookup table to go from tau to ap_ta
        ap=np.logspace(-5, np.log10(self.ap_max), 200)
        tau=[TimeIntegralTa(x,self.w) for x in ap]
        self.ap_root=interp1d(np.log(tau), np.log(ap)) 
    
    def OmegaEvo(self, a):
        OmegaM=1./(1+self.w*a**3-self.wk*a)
        OmegaL=self.w*a**3*OmegaM
        OmegaK=1-OmegaM-OmegaL
        return OmegaM,OmegaL,OmegaK
        
    def HubbleRatioSquare(self, a):
        '''(H(a)/H0)^2'''
        return self.OmegaM0/a/a/a+self.OmegaL0+self.OmegaK0/a/a
    
    def TurnaroundAp(self, tau):
        '''scalefactor (radius) of the perturbation that turns around at time tau'''
        return np.exp(self.ap_root(np.log(tau)))

    def TimeIntegralAp(self, ap, ap_ta, side='left'):
        '''tau=sqrt(OmegaM0)*H0*t, integrated from 0 to ap, for a perturbation which turnaround at ap_ta.
        ap: r/rL0, the radius of the perturbation normalized by its Lagrangian radius at the reference time (a=1).
        ap_ta: specifies the energy parameter (or curvature) of the perturbation, mapped uniquely to a turnaround time a_ta. 
               ap_ta=TurnaroundAp(TimeIntegral(a_ta)).
        side= 'left' or 'right', whether ap is before or after ap_ta.
        '''
        return TimeIntegralAp(ap, ap_ta, self.w, side)
    
    def TimeIntegral(self, a):
        '''tau=sqrt(OmegaM0)*H0*t, at scalefactor a'''
        return TimeIntegral(a, self.w, self.wk)
    
    def Age(self, a):
        '''cosmic time in unit of the Hubble time 1/H0'''
        return self.TimeIntegral(a)/np.sqrt(self.OmegaM0)
    
    def TurnaroundOverdensity(self, a):
        '''delta=rho_ta/rho_bg-1 at scalefactor a of the universe'''
        I=self.TimeIntegral(a)
        ap=self.TurnaroundAp(I)
        return (a/ap)**3-1.
    
    def VirialAp(self, a, return_tau=False):
        '''virial radius in terms of ap: ap_vir=Rvir/R_L0
        if `return_tau`=True, also return the time integral, `\tau`, at `a`.
        '''
        tau=self.TimeIntegral(a)
        ap_ta=self.TurnaroundAp(0.5*tau) #turnaround radius corresponding to the collapse at a. This can also be generalized!!
        ap_vir=ap_ta/2 #virial radius in terms of ap. This may be generalized..
        if return_tau:
            return ap_vir,tau
        else:
            return ap_vir
    
    def VirialDelta(self, a):
        '''DeltaCrit=rho_vir(a)/rho_crit(a)
        DeltaMean=rho_vir(a)/rho_mean(a)
        
        return DeltaCrit,DeltaMean'''
        ap_c=self.VirialAp(a)
        DeltaCrit=self.OmegaM0/self.HubbleRatioSquare(a)/ap_c**3
        DeltaMean=(a/ap_c)**3
        return DeltaCrit,DeltaMean

    def RadiusTimer(self, r, a_ta=1):
        '''orbital time evolution of a shell 
        input:
            r: radius of shell normalized by turnaround radius. r should be a scalar.
            a_ta: scalefactor of the universe when the shell turns around.
        Output:
            t: the time coordinate corresponding to r before turnaround, normalized by turnaround time. The time after turnaround is simply 2-t.
        '''
        tau_ta=self.TimeIntegral(a_ta)
        ap_ta=self.TurnaroundAp(tau_ta)
        return self.TimeIntegralAp(r*ap_ta, ap_ta, 'left')/tau_ta


if __name__=='__main__':
    s=SCSolver(1)
    print("EdS universe:\n delta_ta(a=0.01)=", s.TurnaroundOverdensity(0.01))
    print("Delta_vir(a=0.01)=", s.VirialDelta(1))
    print("Delta_vir(a=1)=", s.VirialDelta(1))

    s=SCSolver(0.268)
    print("LCDM OmegaM=0.268 universe:\n delta_ta(a=0.01)=", s.TurnaroundOverdensity(0.01))
    print("Delta_vir(a=0.01)=", s.VirialDelta(0.01))
    print("Delta_vir(a=1)=", s.VirialDelta(1))
