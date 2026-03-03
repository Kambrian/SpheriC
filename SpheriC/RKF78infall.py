'''7-8 order Runge-Kutta integrator, customized with pericenter counts

Author: 
  Jiaxin Han   
  jiaxin.han@sjtu.edu.cn
  22/09/2021
'''

import numpy as np

class RKF78infall:
    def __init__(self,  x0, t0, rfun, step_max, atol, step_min=1e-8, max_iter=0, verbose=0):
        '''initialize an RKF78 integrator to solve for x(t) with dx/dt=rfun(t,x)
        this function cannot work around the divergence at the origin. It is only applicable to infall motions with non-zero angular momentum, so that the radius is always positive 
                
            x0, t0: initial condition (x0>0 is required for the reflecting boundary to work)
            step_min, step_max: minimum and maximum timestep
            atol: absolute tolerance in x
            max_iter: maximum number of iterations
            verbose: print pericenter crossing events when verbose > 0.
            
            call `self.integrate(tmax)` to integrate.
            the integrated function can be accessed from the `self.t_all`, `self.x_all`, `self.err_all` arrays.
        '''
        self.rfun=rfun
        
        self.hmin=step_min
        self.hmax=step_max
        self.fixstep=(self.hmin==self.hmax)
        self.max_iter=max_iter
        if max_iter:
            self.max_iter=max_iter
        else:
            self.max_iter=np.inf
        
        self.istep=0
        
        self.h=step_max
        
        self.x=np.array(x0)
        self.err=np.zeros_like(self.x)
        
        self.t=t0
        if np.size(atol)>1:
            self.atol=np.array(atol)
        else:
            self.atol=atol*np.ones_like(self.x)
    
        self.x_all=[self.x.copy()]
        self.t_all=[self.t]
        self.err_all=[self.err.copy()]
        
        self.i_peri=0
        self.verbose=verbose
        if x0[0]<0:
                raise RuntimeError("must start from a positive initial position")
        
    def step1(self):
        '''integrate one timestep'''
        #print(self.t,self.h)
        
        #coefficients of RKF78
        k0=self.rfun(self.t, self.x)
        k1=self.rfun(self.t+2./27*self.h, self.x+2./27*k0*self.h)
        k2=self.rfun(self.t+1./9*self.h, self.x+np.dot([1./36,1./12],[k0,k1])*self.h)
        k3=self.rfun(self.t+1./6*self.h, self.x+np.dot([1./24,1./8],[k0,k2])*self.h)
        k4=self.rfun(self.t+5./12*self.h, self.x+np.dot([5./12,-25./16,25./16],[k0,k2,k3])*self.h)
        k5=self.rfun(self.t+1./2*self.h, self.x+np.dot([1./20,1./4,1./5],[k0,k3,k4])*self.h)
        k6=self.rfun(self.t+5./6*self.h, self.x+np.dot([-25./108,125./108,-65./27,125./54],[k0,k3,k4,k5])*self.h)
        k7=self.rfun(self.t+1./6*self.h, self.x+np.dot([31./300,61./225,-2./9,13./900],[k0,k4,k5,k6])*self.h)
        k8=self.rfun(self.t+2./3*self.h, self.x+np.dot([2,-53./6,704./45,-107./9,67./90,3],[k0,k3,k4,k5,k6,k7])*self.h)
        k9=self.rfun(self.t+1./3*self.h, self.x+np.dot([-91./108,23./108,-976./135,311./54,-19./60,17./6,-1./12],[k0,k3,k4,k5,k6,k7,k8])*self.h)
        k10=self.rfun(self.t+self.h, self.x+np.dot([2383./4100,-341./164,4496./1025,-301./82,2133./4100,45./82,45./164,18./41],[k0,k3,k4,k5,k6,k7,k8,k9])*self.h)
        k11=self.rfun(self.t, self.x+np.dot([3./205,-6./41,-3./205,-3./41,3./41,6./41],[k0,k5,k6,k7,k8,k9])*self.h)
        k12=self.rfun(self.t+self.h, self.x+np.dot([-1777./4100,-341./164,4496./1025,-289./82,2193./4100,51./82,33./164,12./41,1.],[k0, k3, k4, k5, k6, k7, k8, k9, k11])*self.h)

        self.err=41./810*(k0+k10-k11-k12)
        if self.fixstep:           #definite step method
            #delta=np.max(abs(41/810*(k0+k10-k11-k12)))
            self.x=self.x+np.dot([34./105,9./35,9./35,9./280,9./280,41./840,41./840],[k5,k6,k7,k8,k9,k11,k12])*self.h
            self.t+=self.h
            self.record()
        else:                    #adaptive step method
            delta=max(np.abs(self.err)/self.atol)
            if delta<=1:            #error under control
                self.x=self.x+np.dot([34./105,9./35,9./35,9./280,9./280,41./840,41./840],[k5,k6,k7,k8,k9,k11,k12])*self.h
                self.t+=self.h
                self.record()
                if delta<0.5 and self.h<self.hmax:      #step too small,begin to amplify step
                    if delta==0:
                        self.h=self.hmax
                    else:
                        self.h=min(self.h*(0.8/delta)**(1./7),self.hmax)
            else:                #error not tolerable, step too big; decrease step and recaculate function value
                if self.h<=self.hmin:
                    print('Error: minimum stepsize exceeded at t=', self.t, ' x=', self.x, ' h=', self.h)
                    raise
                else:
                    self.h=max(self.h*(0.8/delta)**(1./7),self.hmin)
                    self.step1()
                
    
    def detect_pericenter(self):
        '''test and count pericenter passage (if x[0] crosses minimum value)'''
        if (self.x[1]>0) and (self.x_all[-1][1]<=0): #minimum position 
            self.i_peri+=1
            if self.verbose:
                print('pericenter between t=%.2e, dt=%.2e, x=%.2e-%.2e'%(self.t_all[-1], self.h, self.x_all[-1][0], self.x[0]))
                
    def record(self):
        '''record current step'''
        self.detect_pericenter()
        self.x_all.append(self.x.copy())
        self.t_all.append(self.t)
        self.err_all.append(self.err.copy())
        self.istep+=1
        
    def integrate(self, nperi=0, tmax=0):
        '''integrate up to tmax and nperi pericenter passage, whichever is nonzero and tighter'''       
        if nperi==0:
            nperi=np.inf
        if tmax==0:
            tmax=np.inf

        #cast to list for continuation from some previous integration state
        self.t_all=list(self.t_all)
        self.x_all=list(self.x_all)
        self.err_all=list(self.err_all)
        
        while self.t<tmax and self.i_peri<nperi:
            self.step1()
            if self.istep>self.max_iter:
                print('Maximum number of %e iterations exceeded'%self.max_iter, ' t=', self.t, ' stepsize=', self.h)
                break            
        #to array
        self.t_all=np.array(self.t_all)
        self.x_all=np.array(self.x_all)
        self.err_all=np.array(self.err_all)