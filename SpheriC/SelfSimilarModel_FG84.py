'''self similar secondary infall spherical collapse model following Fillmore & Goldreich 1984 and Han 2026

Example usage:
    orb=ReducedOrbit(epsilon=0.3)
    orb.solve()
The mass, density, shell velocity,... can then be accessed through various functions and variables, for example
    x=np.logspace(-2,0)
    #get mass at x:
    orb.GetMass(x)
    orb.GetDensity(x)
    #get reduced time and radius of the orbital evolution:
    orb.tau
    orb.lambd

See the notebook for more examples.

Author: 
  Jiaxin Han   
  jiaxin.han@sjtu.edu.cn
  02/03/2026

Reference: 
    Fillmore & Goldreich 1984
    Han, J. 2026, RAA
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate
#from scipy.misc import derivative
from scipy.signal import argrelextrema

from .RKF78infall import RKF78infall
from .RKF78reflect import RKF78reflect


def intercept(x, y, y0, k=1):
    '''find the location y(x)=y0
    k: order of interpolation method, has to be 1 or 3
    '''
    if k==1:
        dy=y-y0
        indlist=np.where(((dy[:-1]<=0)&(dy[1:]>0))|((dy[:-1]>=0)&(dy[1:]<0)))[0] #bin ids in which [y[i], y[i+1]) contains y0
        x0=(x[indlist+1]-x[indlist])*dy[indlist]/(y[indlist]-y[indlist+1])+x[indlist] #linear interpolated location of x0
    elif k==3:
        dy=scipy.interpolate.UnivariateSpline(x, y-y0, k=3, s=0)
        x0=dy.roots()
    else:
        raise RuntimeError('k=%d not supported in intercept, use 1 or 3'%k)
    return x0

class ReducedOrbit:
    def __init__(self, epsilon=0.3, mu0=lambda x:x, j=1e-2):
        '''object for solving the dimensionless orbit lambda(tau) (Eq 20, 25 of FG84) in 3d (n=3) case

        epsilon: float between [0,1], slope of the initial mass perturbation, delta_i=(M_i/M_0)^-epsilon (Eq-11 of FG84)
        mu0:     function of the initial guess for the reduced mass profile (Eq-21 of FG84)
        j:       J/(r_ta^2/t_ta), reduced angular momentum. set to a tiny non-zero value to avoid divergence at the center.
                 if j==0., however, a special integrator implementing a reflecting boundary will be adopted to solve the orbit.
        #rmin: the minimum radius when evaluating the mass profile
        '''
        self.epsilon=epsilon
        self.e3=1./3./epsilon
        self.mu=mu0 
        self.imu=0 #number of updates in mu
        #self.rmin=rmin
        self.j=j
        
        self.PreCollapse()

    def Lambd_func(self, tau):
        '''Eq-24 of FG84'''
        return tau**(2./3.*(1+self.e3))

    def lambd_tau2(self, tau, lambd):
        '''Eq-25 of FG84'''
        l=np.abs(lambd)
        s=np.sign(lambd) #account for the sign of the acceleration direction
        return -s*np.pi**2/8.*(tau**self.e3/l)**2*self.mu(l/self.Lambd_func(tau))+self.j**2/l**3

    def rfunc(self, tau, y):
        '''vector version of the equation of motion, to be fed to ODE solver
            y: [lambda, lambda_dot]
        return derivative of y over tau
        '''
        return np.array([y[1], self.lambd_tau2(tau, y[0])]) 

    def integrate(self, rconv=1e-3, n_peri_min=10, atol=1e-4, max_timestep=1e-2, **kwargs):
        '''integrate the orbits so that all the orbits reaching out to `r>=rconv*Rta` are resolved (Rta being current turnaround radius).
          using RKF78infall method if j>0, and RKF78reflect if j==0.
            `n_peri_min` specifies the minimum number of pericenter passages to integrate to, in addition to `rconv` requirement.
            `atol`: absolute error in lambda and v_lambda
            `max_timestep`: maximum stepsize in t
            additional kwargs passed to RKF78infall: step_min, max_iter, verbose
        '''
        print('integrating..')
        print('r_apo, n_orbits:')
        if self.j==0: #swith off reflecting boundary in presence of angular momentum
            s=RKF78reflect(x0=[1,0], t0=1, rfun=self.rfunc, step_max=max_timestep, atol=atol, **kwargs)        
        else:
            s=RKF78infall(x0=[1,0], t0=1, rfun=self.rfunc, step_max=max_timestep, atol=atol, **kwargs)
        
        rlast_apo=1.
        laststep=0
        n_peri=n_peri_min+0
        while rlast_apo>rconv: #integrate until the last apocenter is below r_conv, so that mass outside r_conv can all be computed accurately
            s.integrate(n_peri) 
            x=s.x_all.T[0][laststep:s.istep-1]/self.Lambd_func(s.t_all[laststep:s.istep-1])
            rlast_apo=x[np.nonzero(np.r_[True, x[:-1]<=x[1:]]&np.r_[x[:-1]>=x[1:], False])[0][-1]] 
            laststep=s.istep-1
            n_peri*=1.5
            print(rlast_apo, n_peri)
        #self.rmin=rlast_apo #the minimum radius of reliable mass integral
        self.tau=s.t_all
        self.lambd=s.x_all.T[0]
        self.vlambd=s.x_all.T[1] #dlambda/dtau
        self.Lambd=self.Lambd_func(self.tau)
        return self.tau,self.lambd,self.vlambd

    def solve(self, tol_rel=5e-2, rconv=1e-3, max_iter=10, n_peri_min=20, atol=1e-4, plot_on=True):
        '''evolve the system towards convergence
        convergence is reached when the mass profile beyond `rconv` differs less than `tol_rel` between two iterations
        `max_iter`: maximum number of mass profile iterations
        `n_peri_min`: minimum number of periods in orbit integration
        `atol`: absolute tolerance in orbit integration
        `plot_on`: whether to plot the mass profile iterations
        '''
        self.rmin=rconv
        if plot_on:
            x=np.logspace(np.log10(rconv),0,100)
            plt.figure()
            plt.loglog(x, self.mu(x), label='%d'%self.imu)
            plt.xlabel(r'$r/R_{\rm ta}$')
            plt.ylabel(r'$M(<r)/M_{\rm ta}$')
        while True:
            self.integrate(rconv, n_peri_min=n_peri_min, atol=atol, max_timestep=0.01, verbose=0)
            xd,delta=self.update_mu()
            err=np.sqrt(np.mean(delta[xd>rconv]**2))
            print('mass profile iteration %d, relative profile error=%.2e'%(self.imu, err))
            if plot_on:
                #x=np.logspace(np.log10(self.rmin),0,100)
                plt.plot(x, self.mu(x), label='%d, err=%.2f'%(self.imu, err))

            if err<tol_rel:
                break
            if self.imu==max_iter:
                print("maximum number of %d iterations reached when evolving the system, relative error %.2f"%(max_iter,err))
                break
        if plot_on:
            plt.legend()
            plt.show()
            
    def locate_depletion(self, k=1):
        '''get depletion radius and density'''
        self.locate_bounds()
        y=self.lambd/self.Lambd
        self.x_id=y[self.ind_caustic[0]] #r_id/Rta
        self.tau_id=intercept(self.tau, y, self.x_id, k=k)[0]
        self.mu_id=self.tau_id**(-2.*self.e3) #M_id/Mta
        self.rho_id=self.mu_id/self.x_id**3 #rho_id/rho_ta
        self.lambd_id=self.x_id*self.Lambd_func(self.tau_id) 

    def get_mass2(self, x):
        '''alternative method to get_mass(). this is not accurate. deprecated.'''
        h=np.heaviside(x-self.lambd/self.Lambd, 1)
        y=h/self.tau**(1+2*self.e3)
        return 2*self.e3*integrate.trapezoid(y, self.tau)

    def get_mass(self, x, k=1):
        '''the dimensionless mass profile mu(x)=M/M_ta (x) where x=r/R_ta=lambd/Lambd
        k=1 or 3, order of interpolation method. 
        it seems k=3 has some occasional fluctuations, so k=1 preferred.
        (k=1 very slightly higher in the inner part, by 1e-3 factor)
        '''
        if x==1: # to fix possible floating point error in intercept() at the end point
            return 1.
        y=self.lambd/self.Lambd
        if k==1:
            dy=np.diff(y)
            dt=np.diff(self.tau)
            sel_rise=(y[:-1]<=x)&(y[1:]>x)
            sel_fall=(y[:-1]>x)&(y[1:]<=x)
            #apocenter intersections are automatically skipped
            t_rise=self.tau[:-1][sel_rise]+dy[sel_rise]/dt[sel_rise]*(x-y[:-1][sel_rise])
            t_fall=self.tau[:-1][sel_fall]+dt[sel_fall]/dy[sel_fall]*(x-y[:-1][sel_fall])
            m=np.sum(t_fall**(-2.*self.e3))-np.sum(t_rise**(-2.*self.e3))
        else:
            t=intercept(self.tau, y, x, k=k)
            vrat=scipy.interpolate.UnivariateSpline(self.tau, self.vlambd/self.lambd, k=k, s=0)(t)
            signs=-np.sign(vrat-2./3*(1+self.e3)/t)
            m=np.sum(signs*t**(-2.*self.e3))

        return m

    def get_mass_deriv(self,x, k=3):
        '''derivative of mu(x).
        k=1 or 3 specifies the type of interpolation (k=1: linear; k=3: cubic spline) to use when finding the intersecting time of the orbit.
        k=3 leads to more consistent result with the mass profile shape of get_mass().
        k=1 gives a slightly shallower slope in the inner part'''
        if x==1: # to fix possible floating point error in intercept() at the end point
            return 1./(self.epsilon+1./3)
        t=intercept(self.tau, self.lambd/self.Lambd, x, k)
        vrat=scipy.interpolate.UnivariateSpline(self.tau, self.vlambd/self.lambd, k=k, s=0)(t)
        y=1/t**(1+2*self.e3)/x/np.abs(vrat-2./3*(1+self.e3)/t)
        return y.sum()*2*self.e3

    def get_density(self, x, k=3):
        '''rho(x)/rho_ta, at x=r/R_ta'''
        return self.get_mass_deriv(x, k)/3./x/x

    def get_MFR2(self,x, k=3):
        '''MFR/(M_ta/t) at x=r/R_ta.
        alternative (and equivalent) to get_MFR()
        when using k=1, this is slightly noisier than get_MFR(); when using k=3, identical to get_MFR()?
        '''
        t=intercept(self.tau, self.lambd/self.Lambd, x, k)
        vrat=scipy.interpolate.UnivariateSpline(self.tau, self.vlambd/self.lambd, k=k, s=0)(t)
        y=vrat/t**(2*self.e3)/np.abs(vrat-2./3*(1+self.e3)/t)
        return y.sum()*2*self.e3

    def get_MFR(self, x, k=3):
        '''MFR/(M_ta/t) at x=r/R_ta
        k=1 or 3 specifies the type of interpolation (k=1: linear; k=3: cubic spline) to use when finding the intersecting time of the orbit
        different values of k lead to slightly different results in the inner region
        '''
        return 2./3*(1+self.e3)*x*self.get_mass_deriv(x, k)-self.get_mass(x)*2.*self.e3

    def get_vel(self, x):
        '''vmean/(R_ta/t)'''
        return self.get_MFR(x)/self.get_mass_deriv(x)

    def GetMass(self, x):
        '''vector version of get_mass'''
        return np.array(list(map(self.get_mass, x)))

    def GetDensity(self, x):
        '''vector version of get_density'''
        return np.array(list(map(self.get_density, x)))

    def GetMFR(self, x):
        '''vector version of get_MFR'''
        return np.array([self.get_MFR(a) for a in x])

    def GetVel(self, x):
        '''vector version of get_vel'''
        return np.array(list(map(self.get_vel, x)))

    def update_mu(self, n=300):
        x=np.logspace(np.log10(self.rmin), 0, n)
        if x[-1]<1: 
            x.append(1)
        y=np.array([self.get_mass(a) for a in x])
        #y=y/y[-1] #the integrated mass may not equal to 1 but it does not matter much
        y_old=self.mu(x)
        delta=y_old/y-1 
        y=np.log(y)
        f=scipy.interpolate.interp1d(np.log(x), y, bounds_error=False, fill_value='extrapolate') #only linear interp works. higher order cause integration error
        #f=scipy.interpolate.UnivariateSpline(np.log(x), y, k=1, s=0,  ext=0)
        self.mu=lambda x: np.exp(f(np.log(x)))
        #dy=np.diff(y)
        #dx=np.diff(np.log(x))
        #xmid=np.log(x[:-1])+dx/2.
        #f1=scipy.interpolate.interp1d(xmid, dy/dx, bounds_error=False, fill_value='extrapolate')
        #self.mu_deriv=lambda x: self.mu(x)*f1(np.log(x))/x
        #self.mu_deriv=lambda x: self.mu(x)*derivative(f, np.log(x), dx=1e-3)/x

        self.imu+=1
        return x,delta

    def locate_bounds(self):
        '''find indices of peri and apo centers'''
        ind_peri=np.where((self.vlambd[:-1]<0)&(self.vlambd[1:]>0))[0]+1 #out-bound loc of peri
        #ind_apo=np.where((self.vlambd[:-1]>=0)&(self.vlambd[1:]<0))[0] #out-bound loc of apo
        ind_caustic=argrelextrema(self.lambd/self.Lambd, np.greater)[0] #the apocenter at a given time, location of caustics.
        ind_apo=argrelextrema(self.lambd, np.greater)[0] #the apocenter of a given shell
        ind_apo=np.r_[0, ind_apo]
        self.ind_peri=np.array(ind_peri)
        self.ind_caustic=np.array(ind_caustic)
        self.ind_apo=np.array(ind_apo)

    def GetAction(self, iperi):
        '''compute action starting from the iperi-th pericenter till the next pericenter
        call locate_bounds() before computing action.
        '''
        ind0=self.ind_peri[iperi]
        ind1=self.ind_peri[iperi+1]+1
        action=integrate.trapezoid(self.vlambd[ind0:ind1], self.lambd[ind0:ind1])
        return action

    def GetActions(self):
        '''compute actions for all the stored periods'''
        self.locate_bounds()
        self.actions=[self.GetAction(i) for i in range(len(self.ind_peri)-1)]

    def PreCollapse(self, phi_max=np.pi):
        '''orbit before turnaround. classical EdS SC solution without shell crossing.
        phi_max=np.pi will integrate till turnaround
        phi_max=2*np.pi will integrate till collapse (2*turnaround time)
        '''
        phi=np.arange(0.,phi_max,0.1)
        self.tau_pre=(phi-np.sin(phi))/np.pi
        self.lambd_pre=(1-np.cos(phi))/2
        self.vlambd_pre=np.pi/2*np.sin(phi)/(1-np.cos(phi))
        self.Lambd_pre=self.Lambd_func(self.tau_pre)