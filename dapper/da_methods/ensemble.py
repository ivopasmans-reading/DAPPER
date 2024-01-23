"""The EnKF and other ensemble-based methods."""

import numpy as np
import scipy.linalg as sla
from numpy import diag, eye, sqrt, zeros

import dapper.tools.multiproc as multiproc
from dapper.stats import center, inflate_ens, mean0
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, svdi, tinv, tsvd
from dapper.tools.matrices import funm_psd, genOG_1
from dapper.tools.progressbar import progbar
from dapper.tools.randvars import GaussRV
from dapper.tools.seeding import rng
from . import da_method
from abc import ABC, abstractmethod

@da_method
class ens_method:
    """Declare default ensemble arguments."""

    infl: float        = 1.0
    rot: bool          = False
    fnoise_treatm: str = 'Stoch'
    name: str          = ''
            
class EnProcessor:
    """ Class that processes the ensemble and observations. """ 
    
    def __init__(self, **kwargs):
        """ Class constructor. """
        pass 
    
    def pre(self, k, ko, y, E, Y, D):
        """ Function to be called on background ensemble."""
        return E, Y, D 
    
    def assimilate(self, k, ko, y, E, Y, D):
        """ Function carrying out DA correction."""
        return E, Y, D

    def post(self, k, ko, y, E, Y, D):
        """ Function to be called on analysis ensemble. """
        return E, Y, D
    
    def set_hhm(self, HMM):
        """ Link to Hidden Markov model."""
        self.HMM = HMM 
        
    def is_active(self, k, ko):
        """ Indicate whether processor should be used in this step."""
        return ko is not None
    
#----------------------------------------------------------------------
        
class ControlCovariance(EnProcessor):
    """ 
    Calculate cross-covariance between ensemble members
    and predictions in observation space. 
    """
    
    def pre(self, k, ko, y, E, Y, D):
        obs = self.HMM.ObsNow
        Y = obs(E)
        Y = Y - np.mean(Y, axis=0, keepdims=True)
        return E, Y, D

class Inno(EnProcessor):
    """ 
    Transform observation into innovation. 
    """
        
    def pre(self, k, ko, y, E, Y, D):
        #observation operator
        obs = self.HMM.ObsNow
        #Innovations
        D = np.mean(y[None,...] - obs(E), axis=0, keepdims=True) 
        return E, Y, D
    
class StochasticInno(Inno):
    """ 
    Create different innovations for different innovation members. 
    """
    
    def __init__(self, N, **kwargs):
        self.N = N
        if 'No' in kwargs:
            self.N = kwargs['No']
        
    def pre(self, k, ko, y, E, Y, D):
        #observation operator
        obs = self.HMM.ObsNow
        #Errors
        eps = mean0(obs.noise.sample(self.N))
        #Deviations
        Eo = obs(E[:self.N])
        #If number of innovations is larger than ensemble members,
        #bootstrap. 
        if self.N > np.size(E,0):
            Eo = Eo[np.random.randint(0, len(Eo), size=(self.N,))]
        #Calculate innovations.
        D = y[None,...] - eps - Eo        
        return E, Y, D

class Inflator(EnProcessor):
    """
    Processes to maintain ensemble spread.  
    """

class PostInflator(Inflator):
    """ 
    Inflation perturbations from ensemble mean after DA.
    infl=1.0 is no inflation
    """
    
    def __init__(self, infl=1.0, **kwargs):
        if isinstance(infl, (int, float, np.floating, np.integer)):
            self.inflation = lambda k, ko : infl 
        else:
            self.inflation = infl
        
    def post(self, k, ko, y, E, Y, D):
        A, mu = center(E)
        E = mu + A * self.inflation(k,ko)
        return E, Y, D
        
class Rotator(EnProcessor):
    """
    Rotate ensemble around vector 1. 
    """ 
    
    def post(self, k, ko, y, E, Y, D):
        A, mu = center(E)
        N, Nx = E.shape
        T     = eye(N)
        
        T = genOG_1(N, True) @ T
        E = mu + T@A
        
        return E, Y, D
    
#----------------------------------------------------------------------------------------
# Smoothers. Act as pre/post processing to filter step. 
    
class EnStack: 
    """
    First in, first out stack. 
    """
    
    def __init__(self, max_size):
        self.max_size = max_size 
        self.stack, self.times = None, []
        
    def push(self, k, E):  
        """ Push ensemble onto stack. """
        if self.stack is None:
            self.times = [k]
            self.stack = E[None,...]
        elif np.size(self.stack, 0)<self.max_size:
            self.times = np.append(self.times, k)
            self.stack = np.concatenate((self.stack, E[None,...]), axis=0)
        else: 
            self.times = np.roll(self.times, -1, axis=0)
            self.stack = np.roll(self.stack, -1, axis=0)
            self.times[-1] = k
            self.stack[-1] = E     
            
    def top(self):
        """ Display top of stack. """
        return self.times[-1], self.stack[-1] 
    
    def pop(self):
        """ Display and remove top of stack. """
        E = self.stack[-1]
        t = self.times[-1]
        self.stack = self.stack[:-1]
        self.times = self.times[:-1]
        return t, E 
    
    @property 
    def size(self):
        """ Number of ensembles stored in stack. """
        return len(self.times)
    
class AugmentedSmoother(EnProcessor):
    """ 
    Receives an ensemble of states at a single time. Stores 
    it and returns an ensemble of time,space-states.  
    """
    
    def __init__(self, lag, **kwargs):
        super().__init__(**kwargs)
        self.Ea = EnStack(lag+1)
        
    def pre(self, k, ko, y, E, Y, D):
        #Save ensemble
        self.Ea.push(k, E)
        #Transform to time,space-states
        E = self._times2state(self.Ea.stack)
        return E, Y, D
        
    def post(self, k, ko, y, E, Y, D):       
        #Transform back from time,space-states 
        self.Ea.stack = self._state2times(E)   
        #Return latest ensemble.      
        _, E = self.Ea.top()
        return E, Y, D
    
    def _times2state(self, E):
        """ Transform multiple times into 1 state."""
        E = np.transpose(E, [1,0,2])
        E = np.reshape(E, (np.size(E,0),-1))
        return E 
    
    def _state2times(self, E):
        """Split 1 state into multiple times."""
        T = self.Ea.size #number of times in stack
        E = np.reshape(E, (np.size(E,0), T, -1))
        E = np.transpose(E, [1,0,2])
        return E
        
class EnRts(EnProcessor):
    """
    EnRTS (Rauch-Tung-Striebel) smoother.

    Refs: `bib.raanes2016thesis`
    
    Needs to procede an Assimilator object.
    """
    
    def __init__(self, lag, decorr, **kwargs):
        super().__init__(**kwargs)
        #Lag to be stored before backward pass is carried out. 
        self.lag = lag
        #Storage for background and filtered ensemble. 
        self.Ef, self.Ea = EnStack(lag+1), EnStack(lag+1)
        #Decorrelation scale. 
        if isinstance(decorr, (int,np.integer,float,np.floating)):
            decorr = lambda k, ko: decorr
        self.decorr = decorr 
        
    def pre(self, k, ko, y, E, Y, D):    
        self.Ef.push(k, E)
        return E, Y, D
            
    def post(self, k, ko, y, E, Y, D):
        self.Ea.push(k, E)
        
        if self.Ea.size == self.lag+1:
            self._backward_pass(ko)
        
        return E, Y, D
        
    def _backward_pass(self, ko):
        Ef, Ea = self.Ef.stack, self.Ea.stack
        times  = self.Ef.times 
        itimes = range(len(times))
        
        for k in itimes[-1::-1]:
            Aa = center(Ea[k  ])[0]
            Af = center(Ef[k+1])[0]
            
            J  = tinv(Af) @ Aa 
            J *= self.decorr(times[k],ko)
            
            Ea[k] = (Ea[k+1] - Ef[k+1]) @ J
            
    def is_active(self, k, ko):
        #Also needs to work on non-DA steps. 
        return True
    
#----------------------------------------------------------------------

class VaeTransform(EnProcessor):
    """ 
    Use variational autoencoder to transform background and 
    innovations into Latent space. 
    """
    
    def __init__(self, hypermodel, hp, model, **kwargs):
        self.hypermodel  = hypermodel 
        self.hp          = hp
         
        self.ref_model = model 
        self.model     = model  
        self.Elatent   = {'f':[],'a':[]}
        
    def pre(self, k, ko, y, E, Y, D):
        self.train(E, D)
        
        #Convert background ensemble in state space to latent space. 
        _, _, E = self.model.encoder.predict(E, verbose=0) 
        
        #Save for inspection. 
        self.Elatent['f'].append(E)
        
        return E, Y, D
        
    def post(self, k, ko, y, E, Y, D):
        from dapper.vae.basic import rotate        
        #Convert latent background ensemble to state space. 
        Emu, Evar, Esin = self.model.decoder.predict(E, verbose=0) 
        Evar = np.exp(.5*Evar) * np.random.normal(size=np.shape(Evar))    
        E = Emu + rotate(Evar, Esin[:,0])
        E = np.array(E)
        
        #Save for inspection. 
        self.Elatent['a'].append(E)
        
        return E, Y, D
        
    def train(self, E, D):
        pass
    
class CyclingVaeTransform(VaeTransform):
    
    def train(self, E, D):
        hp = self.hp.copy()
        N = np.size(E, 0)
        hp.values['batch_size'] = int(N)
        hp.values['epochs'] = 200
        hp.values['lr_init'] *= .1
        history, self.model = self.hypermodel.fit_bkg(hp, self.model, E)    
        
class BackgroundVaeTransform(VaeTransform):
    
    def train(self, E, D):
        hp = self.hp.copy()
        N = np.size(E, 0)
        hp.values['batch_size'] = int(N)
        hp.values['epochs'] = 200
        hp.values['lr_init'] *= .1
        history, self.model = self.hypermodel.fit_bkg(hp, self.ref_model, E)  
        
class InnoVaeTransform(VaeTransform):    
        
    def pre(self, k, ko, y, E, Y, D):
        self.train(E, D)
            
        #Convert obs-control covariance in observation space. 
        Y = y[None,...] - self.HMM.ObsNow(E)
        _, _, Y = self.model.encoder.predict(Y)
        Y = -Y - np.mean(-Y, axis=0, keepdims=True)
            
        #Convert inno ensemble in state space to latent space. 
        _, _, N = self.model.encoder.predict(D*0)
        _, _, D = self.model.encoder.predict(D)
        D = D - N
        D = np.array(D)
        
        return E, Y, D
    
    def post(self, k, ko, y, E, Y, D):
        return E, Y, D

    def train(self, E, D):
        obs = self.HMM.ObsNow
        N = np.size(E, 0)
        H = obs.linear(np.eye(np.size(E,1)))
        mu = np.mean(E, axis=0)
        
        hp = self.hp.copy()
        hp.values['batch_size'] = int(N)
        hp.values['epochs'] = 200
        hp.values['lr_init'] *= .1 
        hp.values['use_rotation'] = False
        history, self.model = self.hypermodel.fit_obs(hp, self.ref_model, D, 
                                                      H=H, state=mu)
        
#----------------------------------------------------------------------

class Assimilator(EnProcessor):
    """ 
    Base class for all processors that carry out DA update. 
    """
    
    def __init__(self, N, **kwargs):
        self.N  = N
        self.N1 = N-1
    
class EtkfD(Assimilator):
    """ 
    Carry out ETKF using covariance estimated from innovations. 
    """
        
    def assimilate(self, k, ko, y, E, Y, D):
        #Calculate ensemble perturbations. 
        A, Emu = center(E)
        
        #Reshape input. 
        Y, D = Y.T, D.T
        A = A.T
        
        #Covariance of innovations R+HBH
        C = np.cov(D, rowvar=True, ddof=1)
        if np.ndim(C)==0:
            C = np.reshape(C,(1,1))
        Q,L,Qt = np.linalg.svd(np.eye(self.N)-Y.T@np.linalg.pinv(C)@Y / (self.N-1))
        
        #Correction to mean. 
        Kd = A@Y.T@np.linalg.pinv(C)@np.mean(D, axis=1, keepdims=True)/(self.N-1)
        #Correction to ensemble perturburbations.
        A = A@Q@np.diag(np.sqrt(L))@Qt
        
        #Analysis ensemble members. 
        E = Emu[None,...] + A.T + Kd.T
        
        return E, Y, D
    
class PertObs(Assimilator):
    """
    DA using classic, perturbed observations (Burgers'98) 
    """
    
    def assimilate(self, k, ko, y, E, Y, D):        
        R  = self.HMM.ObsNow.noise.C
        A  = E - np.mean(E, axis=0, keepdims=True)
        C  = Y.T @ Y + R * self.N1
        YC = Y@np.linalg.pinv(C)
        KG = A.T @ YC 
        HK = Y.T @ YC 
        dE = (KG @ D.T).T 
        E  = E + dE 
        
        return E, Y, D
    
class SqrtAssimilator(Assimilator):
    """
    Square-root filters. Static methods allow different
    numerical methods to calculate the square-root.  
    """
    
    def __init__(self, solver=None, **kwargs):
        super().__init__(**kwargs)
        
        #Default solver for square-root.
        if solver is None:
            self.solver = SqrtAssimilator.eig_solver 
        else:
            self.solver = solver 
    
    def assimilate(self, k, ko, y, E, Y, D):
        R  = self.HMM.ObsNow.noise.C
        mu = np.mean(E, axis=0, keepdims=True)
        A  = E - mu 
        Pw, T = self.solver(R,Y,D)
        w  = D @ R.inv @ Y.T @ Pw
        HK = R.inv @ Y.T @ Pw @ Y
        E  = mu + w@A + T@A
        return E, Y, D
    
    @staticmethod 
    def explicit_solver(R,Y,D):
        """
        Not recommended due to numerical costs and instability.
        Implementation using inv (in ens space).
        """
        N = np.size(Y,0)
        Pw = sla.inv(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T  = sla.sqrtm(Pw) * sqrt(N-1)
        return Pw, T
    
    @staticmethod 
    def svd_solver(R,Y,D):
        """Implementation using svd of Y R^{-1/2}."""
        N       = np.size(Y,0)
        V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
        d       = pad0(s**2, N) + (N-1)
        Pw      = (V * d**(-1.0)) @ V.T
        T       = (V * d**(-0.5)) @ V.T * sqrt(N-1)
        return Pw, T
        
    @staticmethod 
    def ss_solver(R,Y,D):
        """ 
        Same as 'svd', but with slightly different notation
        (sometimes used by Sakov) using the normalization sqrt(N1).
        """
        N       = np.size(Y,0)
        S       = Y @ R.sym_sqrt_inv.T / sqrt(N-1)
        V, s, _ = svd0(S)
        d       = pad0(s**2, N) + 1
        Pw      = (V * d**(-1.0))@V.T / (N-1)  # = G/(N1)
        T       = (V * d**(-0.5))@V.T
        return Pw, T
    
    @staticmethod 
    def eig_solver(R,Y,D):
        """
        'eig' in upd_a:
        Implementation using eig. val. decomp.
        """
        N      = np.size(Y,0)
        d, V   = sla.eigh(Y @ R.inv @ Y.T + (N-1)*eye(N))
        T      = V@diag(d**(-0.5))@V.T * sqrt(N-1)
        Pw     = V@diag(d**(-1.0))@V.T
        return Pw, T
  
class SerialAssimilator(Assimilator,ABC):  
    """ 
    Updating ensemble after assimilation of each observations. 
    IMPORTANT: this Assimilator also update Y and D. 
    """
        
    def __call__(self, k, ko, y, E, Y, D):
        #Observation covariance matrix
        R = self.HMM.ObsNow.noise.C
        #Ensemble perturbations 
        mu = np.mean(E, axis=0, keepdims=True)
        A  = E - mu
        # Observations assimilated one-at-a-time:
        inds = self.sorter(y, R, A)
        #Requires de-correlation:
        D = D @ R.sym_sqrt_inv.T
        Y = Y @ R.sym_sqrt_inv.T
        # Carry out actual DA
        E,Y,D = self.assimilate(inds,mu,A,Y,D)
        #Recorrelated
        Y = Y @ R.sym_sqrt.T 
        D = D @ R.sym_sqrt.T
        
        return E, Y, D
    
    @staticmethod 
    def mono_sorter(y, R, A):
        return np.range(len(y))
    
    @staticmethod 
    def var_sorter(y, R, A):
        N = len(A)
        dC = R.diag
        if np.all(dC == dC[0]):
            # Sort y by P
            dC = np.sum(A*A, 0)/(N-1)
        inds = np.argsort(dC)
        
    @staticmethod 
    def random_sorter(y, R, A):
        return rng.permutation(len(y))

class PerturbedSerialAssimilator(SerialAssimilator):
    
    def __init__(self, sorter=None, perturber=None, **kwargs):
        super().__init__(**kwargs)
        
        #Set way to sort processing order observations.
        if sorter is None:
            self.sorter = SerialAssimilator.random_sorter
        else:
            self.sorter = sorter 
        
        #How to perturbe observations. 
        if perturber is None:
            self.perturber = PerturbedSerialAssimilator.stoch_perturber
        else:
            self.perturber = perturber
            
    def assimilate(self,inds,mu,A,Y,D):
        # Enhancement in the nonlinear case:
        # re-compute Y each scalar obs assim.
        # But: little benefit, model costly (?),
        # updates cannot be accumulated on S and T.

        # More details: Misc/Serial_ESOPS.py.
        for i, j in enumerate(inds):
            Zj = self.perturber(i,j,A)

            # Select j-th obs
            Yj  = Y[:, j]       # [j] obs anomalies
            dyj = D[0, j]         # [j] innov mean
            DYj = Zj - Yj       # [j] innov anomalies
            DYj = DYj[:, None]  # Make 2d vertical

            # Kalman gain computation
            C     = Yj@Yj + self.N-1  # Total obs cov
            KGx   = Yj @ A / C  # KG to update state
            KGy   = Yj @ Y / C  # KG to update obs

            # Updates
            A    += DYj * KGx
            mu   += dyj * KGx
            Y    += DYj * KGy
            D[0] -= dyj * KGy
            
        #Update ensemble members. 
        E = mu + A
        return E, Y, D  
    
    @staticmethod 
    def stoch_perturber(i,j,A):
        # The usual stochastic perturbations.
        N = np.size(A,0)
        Zj = mean0(rng.standard_normal(N))  # Un-coloured noise
        return Zj
        
    @staticmethod 
    def norm_stoch_perturber(i,j,A):
        N = np.size(A,0)
        Zj  = PerturbedSerialAssimilator.stoch_perturber(i,j,A)
        Zj *= sqrt(N/(Zj@Zj))
        return Zj 
    
    @staticmethod 
    def esops_perturber(i,j,A):
        # "2nd-O exact perturbation sampling"
        N = np.size(A,0)
        if i == 0:
            # Init -- increase nullspace by 1
            V, s, UT = svd0(A)
            s[N-2:] = 0
            A = svdi(V, s, UT)
            v = V[:, N-2]
        else:
            raise RuntimeError('Code missing: no variable v.')
            # Orthogonalize v wrt. the new A
            #
            # v = Zj - Yj (from paper) requires Y==HX.
            # Instead: mult` should be c*ones(Nx) so we can
            # project v into ker(A) such that v@A is null.
            mult  = (v@A) / (Yj@A) # noqa
            v     = v - mult[0]*Yj # noqa
            v    /= sqrt(v@v)
        
        Zj    = v*sqrt(N-1)  # Standardized perturbation along v
        Zj   *= np.sign(rng.standard_normal() - 0.5)  # Random sign
        return Zj
    
class EnSrf(SerialAssimilator):
    """
    Potter scheme, "EnSRF"
    - EAKF's two-stage "update-regress" form yields
      the same *ensemble* as this.
    - The form below may be derived as "serial ETKF",
      but does not yield the same
      ensemble as 'Sqrt' (which processes obs as a batch)
      -- only the same mean/cov.
    """
    
    def __init__(self, sorter=None, **kwargs):
        super().__init__(**kwargs)
        
        #Set way to sort processing order observations.
        if sorter is None:
            self.sorter = SerialAssimilator.random_sorter
        else:
            self.sorter = sorter 
      
    def assimilate(self, inds, mu, A, Y, D):
        T  = eye(self.N)
        for j in inds:
            Yj = Y[:, j]
            C  = Yj@Yj + self.N1
            Tj = np.outer(Yj, Yj / (C + sqrt(self.N1*C)))
            T -= Tj @ T
            
            Y -= Tj @ Y
            
        w = D@Y.T@T/self.N1
        E = mu + w@A + T@A
        return E, Y, D
        
class Denkf(Assimilator):
    """ 
    Uses "Deterministic EnKF" (sakov'08)
    """
    
    def assimilate(self, k, ko, y, E, Y, D):
        A  = E - np.mean(E, axis=0, keepdims=True)
        d  = np.mean(D,axis=0)
        R  = self.HMM.ObsNow.noise.C
        C  = Y.T @ Y + R.full*self.N1
        YC = Y@np.linalg.pinv(C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@d - 0.5*(KG@Y.T).T
        return E, Y, D
        
        
        
    
#----------------------------------------------------------------------

@ens_method        
class EnDa:
    """ 
    General class for ensemble Kalman filters/smoothers 
    """ 
    N: int 
    processors: list
   
    def assimilate(self, HMM, xx, yy):
        # Init
        for processor in self.processors:
            processor.set_hhm(HMM)
            
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)
        
        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)
            
            #Logging if analysis update
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                HMM.ObsNow = HMM.Obs(ko)
                yyNow = yy[ko]
            else:
                yyNow = None
               
            #Select processors to be used in this step.  
            processors = [process for process in self.processors 
                          if process.is_active(k,ko)]
            
            D, Y = [], []
            for process in processors[ 0:: 1]:
                E, Y, D = process.pre(k, ko, yyNow, E, Y, D)
            for process in processors[ 0:: 1]:
                E, Y, D = process.assimilate(k, ko, yyNow, E, Y, D)
            for process in processors[-1::-1]:
                E, Y, D = process.post(k, ko, yyNow, E, Y, D)

            if ko is not None:
                self.stats.assess(k, ko, E=E)
   
   
class EndaFactory:
    
    def build(self, N, filter='', smoother='', **kwargs):
        self.options = {**kwargs, 'N':N,
                        'filter':str.lower(filter), 
                        'smoother':str.lower(smoother)}     

        processors = []
        processors = self.create_D_builder(processors)
        processors = self.create_Y_builder(processors)
        processors = self.create_modifications(processors)
        processors = self.create_smoother(processors)
        processors = self.create_filter(processors)
        
        #Set attributes inherited from ens_method. 
        enda = EnDa(N=self.options['N'], processors=processors)
        for key in set(self.options).intersection(dir(enda)):
            setattr(enda, key, self.options[key])
        
        return enda
        
    def create_D_builder(self, processors):
        filter = self.options['filter']  
        
        if 'pertobs' in filter or 'etkf_d' in filter:
            processors += [StochasticInno(**self.options)]
        else:
            processors += [Inno(**self.options)]  
            
        return processors 
            
    def create_Y_builder(self, processors):
        #Create obs-controlcovariance 
        if 'VaeTransforms' in self.options:
            processors += self.options['VaeTransforms']     
        else:
            processors += [ControlCovariance()] 
              
        return processors
            
    def create_modifications(self, processors):
        if 'rot' in self.options and self.options['rot'] is True:
            processors += [Rotator(**self.options)]
        if 'infl' in self.options:
            processors += [PostInflator(**self.options)]
        
        return processors  
    
    def create_smoother(self, processors):
        smoother = self.options['smoother']
        
        if 'enks' in smoother:
            processors += [AugmentedSmoother(**self.options)]
        elif 'ensrf' in smoother:
            processors += [EnRTS(**self.options)]
            
        return processors
            
    def create_filter(self, processors):
        filter = self.options['filter']
        
        if 'sqrt' in filter:
            processors  = self._create_sqrt_filter(processors)
        elif 'serial' in filter:
            processors  = self._create_serial_filter(processors)
        elif 'denkf' in filter:
            processors += [Denkf(**self.options)]
        elif 'etkf_d' in filter:
            processors += [EtkfD(**self.options)]
            
        return processors 
    
    def _create_sqrt_filter(self, processors):
        filter = self.options['filter']
        
        #Sqrt root filter
        if 'explicit' in filter:
            solver = SqrtAssimilator.explicit_solver 
        elif 'svd' in filter:
            solver = SqrtAssimilator.svd_solver 
        elif 'ss' in filter:
            solver = SqrtAssimilator.ss_solver 
        else:
            solver = SqrtAssimilator.eig_solver  
        
        processors += [SqrtAssimilator(solver=solver, **self.options)]
        return processors 
    
    def _create_serial_filter(self, processors):
        filter = self.options['filter']
        
        if 'stoch' in filter and 'var1' in filter:
            perturber = PerturbedSerialAssimilator.norm_stoch_perturber 
        elif 'stoch' in filter:
            perturber = PerturbedSerialAssimilator.stoch_perturber 
        elif 'esops' in filter: 
            perturber = PerturbedSerialAssimilator.esops_perturber
        else:
            perturber = None 
            
        if 'mono' in filter:
            sorter = SerialAssimilator.mono_sorter 
        elif 'sorted' in filter: 
            sorter = SerialAssimilator.var_sorter
        else:
            sorter = SerialAssimilator.random_sorter
            
        if perturber is None:
            processors += [EnSrf(sorter=sorter,**self.options)]
        else:
            processors += [PerturbedSerialAssimilator(perturber=perturber, 
                                                      sorter=sorter,
                                                      **self.options)]
        
        return processors
            
#----------------------------------------------------------------------
         
@ens_method
class EnKF:
    """The ensemble Kalman filter.

    Refs: `bib.evensen2009ensemble`.
    """

    upd_a: str
    N: int

    def assimilate(self, HMM, xx, yy):
        # Init
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            # Analysis update
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                E = EnKF_analysis(E, HMM.Obs(ko)(E), HMM.Obs(ko).noise, yy[ko],
                                  self.upd_a, self.stats, ko)
                E = post_process(E, self.infl, self.rot)

            self.stats.assess(k, ko, E=E)

def EnKF_analysis(E, Eo, hnoise, y, upd_a, stats=None, ko=None):
    """Perform the EnKF analysis update.

    This implementation includes several flavours and forms,
    specified by `upd_a`.

    Main references: `bib.sakov2008deterministic`,
    `bib.sakov2008implications`, `bib.hoteit2015mitigating`
    """
    R     = hnoise.C     # Obs noise cov
    N, Nx = E.shape      # Dimensionality
    N1    = N-1          # Ens size - 1

    mu = np.mean(E, 0)   # Ens mean
    A  = E - mu          # Ens anomalies

    xo = np.mean(Eo, 0)  # Obs ens mean
    Y  = Eo-xo           # Obs ens anomalies
    dy = y - xo          # Mean "innovation"

    if 'PertObs' in upd_a:
        # Uses classic, perturbed observations (Burgers'98)
        C  = Y.T @ Y + R.full*N1
        D  = mean0(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ (y - D - Eo).T).T
        E  = E + dE

    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.

        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == 'Sqrt' and N > Nx:
            upd_a = 'Sqrt svd'

        if 'explicit' in upd_a:
            # Not recommended due to numerical costs and instability.
            # Implementation using inv (in ens space)
            Pw = sla.inv(Y @ R.inv @ Y.T + N1*eye(N))
            T  = sla.sqrtm(Pw) * sqrt(N1)
            HK = R.inv @ Y.T @ Pw @ Y
            # KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
            # Implementation using svd of Y R^{-1/2}.
            V, s, _ = svd0(Y @ R.sym_sqrt_inv.T)
            d       = pad0(s**2, N) + N1
            Pw      = (V * d**(-1.0)) @ V.T
            T       = (V * d**(-0.5)) @ V.T * sqrt(N1)
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2+N1)**(-1.0) * s**2)
        elif 'sS' in upd_a:
            # Same as 'svd', but with slightly different notation
            # (sometimes used by Sakov) using the normalization sqrt(N1).
            S       = Y @ R.sym_sqrt_inv.T / sqrt(N1)
            V, s, _ = svd0(S)
            d       = pad0(s**2, N) + 1
            Pw      = (V * d**(-1.0))@V.T / N1  # = G/(N1)
            T       = (V * d**(-0.5))@V.T
            # docs/snippets/trHK.jpg
            trHK    = np.sum((s**2 + 1)**(-1.0)*s**2)
        else:  # 'eig' in upd_a:
            # Implementation using eig. val. decomp.
            d, V   = sla.eigh(Y @ R.inv @ Y.T + N1*eye(N))
            T      = V@diag(d**(-0.5))@V.T * sqrt(N1)
            Pw     = V@diag(d**(-1.0))@V.T
            HK     = R.inv @ Y.T @ (V @ diag(d**(-1)) @ V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A

    elif 'Serial' in upd_a:
        # Observations assimilated one-at-a-time:
        inds = serial_inds(upd_a, y, R, A)
        #  Requires de-correlation:
        dy   = dy @ R.sym_sqrt_inv.T
        Y    = Y  @ R.sym_sqrt_inv.T
        # Enhancement in the nonlinear case:
        # re-compute Y each scalar obs assim.
        # But: little benefit, model costly (?),
        # updates cannot be accumulated on S and T.

        if any(x in upd_a for x in ['Stoch', 'ESOPS', 'Var1']):
            # More details: Misc/Serial_ESOPS.py.
            for i, j in enumerate(inds):

                # Perturbation creation
                if 'ESOPS' in upd_a:
                    # "2nd-O exact perturbation sampling"
                    if i == 0:
                        # Init -- increase nullspace by 1
                        V, s, UT = svd0(A)
                        s[N-2:] = 0
                        A = svdi(V, s, UT)
                        v = V[:, N-2]
                    else:
                        # Orthogonalize v wrt. the new A
                        #
                        # v = Zj - Yj (from paper) requires Y==HX.
                        # Instead: mult` should be c*ones(Nx) so we can
                        # project v into ker(A) such that v@A is null.
                        mult  = (v@A) / (Yj@A) # noqa
                        v     = v - mult[0]*Yj # noqa
                        v    /= sqrt(v@v)
                    Zj  = v*sqrt(N1)  # Standardized perturbation along v
                    Zj *= np.sign(rng.standard_normal() - 0.5)  # Random sign
                else:
                    # The usual stochastic perturbations.
                    Zj = mean0(rng.standard_normal(N))  # Un-coloured noise
                    if 'Var1' in upd_a:
                        Zj *= sqrt(N/(Zj@Zj))

                # Select j-th obs
                Yj  = Y[:, j]       # [j] obs anomalies
                dyj = dy[j]         # [j] innov mean
                DYj = Zj - Yj       # [j] innov anomalies
                DYj = DYj[:, None]  # Make 2d vertical

                # Kalman gain computation
                C     = Yj@Yj + N1  # Total obs cov
                KGx   = Yj @ A / C  # KG to update state
                KGy   = Yj @ Y / C  # KG to update obs

                # Updates
                A    += DYj * KGx
                mu   += dyj * KGx
                Y    += DYj * KGy
                dy   -= dyj * KGy
            E = mu + A
        else:
            # "Potter scheme", "EnSRF"
            # - EAKF's two-stage "update-regress" form yields
            #   the same *ensemble* as this.
            # - The form below may be derived as "serial ETKF",
            #   but does not yield the same
            #   ensemble as 'Sqrt' (which processes obs as a batch)
            #   -- only the same mean/cov.
            T = eye(N)
            for j in inds:
                Yj = Y[:, j]
                C  = Yj@Yj + N1
                Tj = np.outer(Yj, Yj / (C + sqrt(N1*C)))
                T -= Tj @ T
                Y -= Tj @ Y
            w = dy@Y.T@T/N1
            E = mu + w@A + T@A

    elif 'DEnKF' == upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.full*N1
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T

    else:
        raise KeyError("No analysis update method found: '" + upd_a + "'.")

    # Diagnostic: relative influence of observations
    if stats is not None:
        if 'trHK' in locals():
            stats.trHK[ko] = trHK      / hnoise.M
        elif 'HK' in locals():
            stats.trHK[ko] = HK.trace()/hnoise.M

    return E


def post_process(E, infl, rot):
    """Inflate, Rotate.

    To avoid recomputing/recombining anomalies,
    this should have been inside `EnKF_analysis`

    But it is kept as a separate function

    - for readability;
    - to avoid inflating/rotationg smoothed states (for the `EnKS`).
    """
    do_infl = infl != 1.0 and infl != '-N'

    if do_infl or rot:
        A, mu  = center(E)
        N, Nx  = E.shape
        T      = eye(N)

        if do_infl:
            T = infl * T

        if rot:
            T = genOG_1(N, rot) @ T

        E = mu + T@A
    return E


def add_noise(E, dt, noise, method):
    """Treatment of additive noise for ensembles.

    Refs: `bib.raanes2014ext`
    """
    if noise.C == 0:
        return E

    N, Nx = E.shape
    A, mu = center(E)
    Q12   = noise.C.Left
    Q     = noise.C.full

    def sqrt_core():
        T    = np.nan    # cause error if used
        Qa12 = np.nan    # cause error if used
        A2   = A.copy()  # Instead of using (the implicitly nonlocal) A,
        # which changes A outside as well. NB: This is a bug in Datum!
        if N <= Nx:
            Ainv = tinv(A2.T)
            Qa12 = Ainv@Q12
            T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), sqrt)
            A2   = T@A2
        else:  # "Left-multiplying" form
            P  = A2.T @ A2 / (N-1)
            L  = funm_psd(eye(Nx) + dt*mrdiv(Q, P), sqrt)
            A2 = A2 @ L.T
        E = mu + A2
        return E, T, Qa12

    if method == 'Stoch':
        # In-place addition works (also) for empty [] noise sample.
        E += sqrt(dt)*noise.sample(N)

    elif method == 'none':
        pass

    elif method == 'Mult-1':
        varE   = np.var(E, axis=0, ddof=1).sum()
        ratio  = (varE + dt*diag(Q).sum())/varE
        E      = mu + sqrt(ratio)*A
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Mult-M':
        varE   = np.var(E, axis=0)
        ratios = sqrt((varE + dt*diag(Q))/varE)
        E      = mu + A*ratios
        E      = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Core':
        E = sqrt_core()[0]

    elif method == 'Sqrt-Mult-1':
        varE0 = np.var(E, axis=0, ddof=1).sum()
        varE2 = (varE0 + dt*diag(Q).sum())
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            A, mu   = center(E)
            varE1   = np.var(E, axis=0, ddof=1).sum()
            ratio   = varE2/varE1
            E       = mu + sqrt(ratio)*A
            E       = svdi(*tsvd(E, 0.999))  # Explained in Datum

    elif method == 'Sqrt-Add-Z':
        E, _, Qa12 = sqrt_core()
        if N <= Nx:
            Z  = Q12 - A.T@Qa12
            E += sqrt(dt)*(Z@rng.standard_normal((Z.shape[1], N))).T

    elif method == 'Sqrt-Dep':
        E, T, Qa12 = sqrt_core()
        if N <= Nx:
            # Q_hat12: reuse svd for both inversion and projection.
            Q_hat12      = A.T @ Qa12
            U, s, VT     = tsvd(Q_hat12, 0.99)
            Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
            Q_hat12_proj = VT.T@VT
            rQ = Q12.shape[1]
            # Calc D_til
            Z      = Q12 - Q_hat12
            D_hat  = A.T@(T-eye(N))
            Xi_hat = Q_hat12_inv @ D_hat
            Xi_til = (eye(rQ) - Q_hat12_proj)@rng.standard_normal((rQ, N))
            D_til  = Z@(Xi_hat + sqrt(dt)*Xi_til)
            E     += D_til.T

    else:
        raise KeyError('No such method')

    return E



@ens_method
class EnKS:
    """The ensemble Kalman smoother.

    Refs: `bib.evensen2009ensemble`

    The only difference to the EnKF
    is the management of the lag and the reshapings.
    """

    upd_a: str
    N: int
    Lag: int

    # Reshapings used in smoothers to go to/from
    # 3D arrays, where the 0th axis is the Lag index.
    def reshape_to(self, E):
        K, N, Nx = E.shape
        return E.transpose([1, 0, 2]).reshape((N, K*Nx))

    def reshape_fr(self, E, Nx):
        N, Km = E.shape
        K    = Km//Nx
        return E.reshape((N, K, Nx)).transpose([1, 0, 2])

    def assimilate(self, HMM, xx, yy):
        # Inefficient version, storing full time series ensemble.
        # See iEnKS for a "rolling" version.
        E    = zeros((HMM.tseq.K+1, self.N, HMM.Dyn.M))
        E[0] = HMM.X0.sample(self.N)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E[k] = HMM.Dyn(E[k-1], t-dt, dt)
            E[k] = add_noise(E[k], dt, HMM.Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E[k])

                Eo    = HMM.Obs(ko)(E[k])
                y     = yy[ko]

                # Inds within Lag
                kk    = range(max(0, k-self.Lag*HMM.tseq.dko), k+1)

                EE    = E[kk]

                EE    = self.reshape_to(EE)
                EE    = EnKF_analysis(EE, Eo, HMM.Obs(ko).noise, y,
                                      self.upd_a, self.stats, ko)
                E[kk] = self.reshape_fr(EE, HMM.Dyn.M)
                E[k]  = post_process(E[k], self.infl, self.rot)
                self.stats.assess(k, ko, 'a', E=E[k])

        for k, ko, _, _ in progbar(HMM.tseq.ticker, desc='Assessing'):
            self.stats.assess(k, ko, 'u', E=E[k])
            if ko is not None:
                self.stats.assess(k, ko, 's', E=E[k])


@ens_method
class EnRTS:
    """EnRTS (Rauch-Tung-Striebel) smoother.

    Refs: `bib.raanes2016thesis`
    """

    upd_a: str
    N: int
    DeCorr: float

    def assimilate(self, HMM, xx, yy):
        E    = zeros((HMM.tseq.K+1, self.N, HMM.Dyn.M))
        Ef   = E.copy()
        E[0] = HMM.X0.sample(self.N)

        # Forward pass
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E[k]  = HMM.Dyn(E[k-1], t-dt, dt)
            E[k]  = add_noise(E[k], dt, HMM.Dyn.noise, self.fnoise_treatm)
            Ef[k] = E[k]

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E[k])
                Eo   = HMM.Obs(ko)(E[k])
                y    = yy[ko]
                E[k] = EnKF_analysis(E[k], Eo, HMM.Obs(ko).noise, y,
                                     self.upd_a, self.stats, ko)
                E[k] = post_process(E[k], self.infl, self.rot)
                self.stats.assess(k, ko, 'a', E=E[k])

        # Backward pass
        for k in progbar(range(HMM.tseq.K)[::-1]):
            A  = center(E[k])[0]
            Af = center(Ef[k+1])[0]

            J = tinv(Af) @ A
            J *= self.DeCorr

            E[k] += (E[k+1] - Ef[k+1]) @ J

        for k, ko, _, _ in progbar(HMM.tseq.ticker, desc='Assessing'):
            self.stats.assess(k, ko, 'u', E=E[k])
            if ko is not None:
                self.stats.assess(k, ko, 's', E=E[k]) 
        
def serial_inds(upd_a, y, cvR, A):
    """Get the indices used for serial updating.

    - Default: random ordering
    - if "mono" in `upd_a`: `1, 2, ..., len(y)`
    - if "sorted" in `upd_a`: sort by variance
    """
    if 'mono' in upd_a:
        # Not robust?
        inds = np.arange(len(y))
    elif 'sorted' in upd_a:
        N = len(A)
        dC = cvR.diag
        if np.all(dC == dC[0]):
            # Sort y by P
            dC = np.sum(A*A, 0)/(N-1)
        inds = np.argsort(dC)
    else:  # Default: random ordering
        inds = rng.permutation(len(y))
    return inds


@ens_method
class SL_EAKF:
    """Serial, covariance-localized EAKF.

    Refs: `bib.karspeck2007experimental`.

    In contrast with LETKF, this iterates over the observations rather
    than over the state (batches).

    Used without localization, this should be equivalent (full ensemble equality)
    to the `EnKF` with `upd_a='Serial'`.
    """

    N: int
    loc_rad: float
    taper: str  = 'GC'
    ordr: str   = 'rand'

    def assimilate(self, HMM, xx, yy):
        N1   = self.N-1

        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                Obs  = HMM.Obs(ko)
                R    = Obs.noise
                y    = yy[ko]
                inds = serial_inds(self.ordr, y, R, center(E)[0])
                Rm12 = Obs.noise.C.sym_sqrt_inv

                state_taperer = Obs.localizer(self.loc_rad, 'y2x', self.taper)
                for j in inds:
                    # Prep:
                    # ------------------------------------------------------
                    Eo = Obs(E)
                    xo = np.mean(Eo, 0)
                    Y  = Eo - xo
                    mu = np.mean(E, 0)
                    A  = E-mu
                    # Update j-th component of observed ensemble:
                    # ------------------------------------------------------
                    Y_j    = Rm12[j, :] @ Y.T
                    dy_j   = Rm12[j, :] @ (y - xo)
                    # Prior var * N1:
                    sig2_j = Y_j@Y_j
                    if sig2_j < 1e-9:
                        continue
                    # Update (below, we drop the locality subscript: _j)
                    sig2_u = 1/(1/sig2_j + 1/N1)      # Postr. var * N1
                    alpha  = (N1/(N1+sig2_j))**(0.5)  # Update contraction factor
                    dy2    = sig2_u * dy_j/N1         # Mean update
                    Y2     = alpha*Y_j                # Anomaly update
                    # Update state (regress update from obs space, using localization)
                    # ------------------------------------------------------
                    ii, tapering = state_taperer(j)
                    # ii, tapering = ..., 1  # cancel localization
                    if len(ii) == 0:
                        continue
                    Xi = A[:, ii]*tapering
                    Regression = Xi.T @ Y_j/np.sum(Y_j**2)
                    mu[ii] += Regression*dy2
                    A[:, ii] += np.outer(Y2 - Y_j, Regression)
                    E = mu + A

                E = post_process(E, self.infl, self.rot)

            self.stats.assess(k, ko, E=E)


def local_analyses(E, Eo, R, y, state_batches, obs_taperer, mp=map, xN=None, g=0):
    """Perform local analysis update for the LETKF."""
    def local_analysis(ii):
        """Perform analysis, for state index batch `ii`."""
        # Locate local domain
        oBatch, tapering = obs_taperer(ii)
        Eii = E[:, ii]

        # No update
        if len(oBatch) == 0:
            return Eii, 1

        # Localize
        Yl  = Y[:, oBatch]
        dyl = dy[oBatch]
        tpr = sqrt(tapering)

        # Adaptive inflation estimation.
        # NB: Localisation is not 100% compatible with the EnKF-N, since
        # - After localisation there is much less need for inflation.
        # - Tapered values (Y, dy) are too neat
        #   (the EnKF-N expects a normal amount of sampling error).
        # One fix is to tune xN (maybe set it to 2 or 3). Thanks to adaptivity,
        # this should still be easier than tuning the inflation factor.
        infl1 = 1 if xN is None else sqrt(N1/effective_N(Yl, dyl, xN, g))
        Eii, Yl = inflate_ens(Eii, infl1), Yl * infl1
        # Since R^{-1/2} was already applied (necesry for effective_N), now use R=Id.
        # TODO 4: the cost of re-init this R might not always be insignificant.
        R = GaussRV(C=1, M=len(dyl))

        # Update
        Eii = EnKF_analysis(Eii, Yl*tpr, R, dyl*tpr, "Sqrt")

        return Eii, infl1

    # Prepare analysis
    N1 = len(E) - 1
    Y, xo = center(Eo)
    # Transform obs space
    Y  = Y        @ R.sym_sqrt_inv.T
    dy = (y - xo) @ R.sym_sqrt_inv.T

    # Run
    result = mp(local_analysis, state_batches)

    # Assign
    E_batches, infl1 = zip(*result)
    # TODO: this overwrites E, possibly unbeknownst to caller
    for ii, Eii in zip(state_batches, E_batches):
        E[:, ii] = Eii

    return E, dict(ad_inf=sqrt(np.mean(np.array(infl1)**2)))


@ens_method
class LETKF:
    """Same as EnKF (Sqrt), but with localization.

    Refs: `bib.hunt2007efficient`.

    NB: Multiproc. yields slow-down for `dapper.mods.Lorenz96`,
    even with `batch_size=(1,)`. But for `dapper.mods.QG`
    (`batch_size=(2,2)` or less) it is quicker.

    NB: If `len(ii)` is small, analysis may be slowed-down with '-N' infl.
    """

    N: int
    loc_rad: float
    taper: str = 'GC'
    xN: float  = None
    g: int     = 0
    mp: bool   = False

    def assimilate(self, HMM, xx, yy):
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)
        self.stats.new_series("ad_inf", 1, HMM.tseq.Ko+1)

        with multiproc.Pool(self.mp) as pool:
            for k, ko, t, dt in progbar(HMM.tseq.ticker):
                E = HMM.Dyn(E, t-dt, dt)
                E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

                if ko is not None:
                    self.stats.assess(k, ko, 'f', E=E)
                    Obs = HMM.Obs(ko)
                    batch, taper = Obs.localizer(self.loc_rad, 'x2y', self.taper)
                    E, stats = local_analyses(E, Obs(E), Obs.noise.C, yy[ko],
                                              batch, taper, pool.map, self.xN, self.g)
                    self.stats.write(stats, k, ko, "a")
                    E = post_process(E, self.infl, self.rot)

                self.stats.assess(k, ko, E=E)


def effective_N(YR, dyR, xN, g):
    """Effective ensemble size N.

    As measured by the finite-size EnKF-N
    """
    N, Ny = YR.shape
    N1   = N-1

    V, s, UT = svd0(YR)
    du     = UT @ dyR

    eN, cL = hyperprior_coeffs(s, N, xN, g)

    def pad_rk(arr): return pad0(arr, min(N, Ny))
    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

    # Make dual cost function (in terms of l1)
    def J(l1):
        val = np.sum(du**2/dgn_rk(l1)) \
            + eN/l1**2 \
            + cL*np.log(l1**2)
        return val

    # Derivatives (not required with minimize_scalar):
    def Jp(l1):
        val = -2*l1   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
            + -2*eN/l1**3 \
            + 2*cL/l1
        return val

    def Jpp(l1):
        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
            + 6*eN/l1**4 \
            + -2*cL/l1**2
        return val

    # Find inflation factor (optimize)
    l1 = Newton_m(Jp, Jpp, 1.0)
    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x

    za = N1/l1**2
    return za


# Notes on optimizers for the 'dual' EnKF-N:
# ----------------------------------------
#  Using minimize_scalar:
#  - Doesn't take dJdx. Advantage: only need J
#  - method='bounded' not necessary and slower than 'brent'.
#  - bracket not necessary either...
#  Using multivariate minimization: fmin_cg, fmin_bfgs, fmin_ncg
#  - these also accept dJdx. But only fmin_bfgs approaches
#    the speed of the scalar minimizers.
#  Using scalar root-finders:
#  - brenth(dJ1, LowB, 1e2,     xtol=1e-6) # Same speed as minimization
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6) # No improvement
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6, fprime2=dJ3) # No improvement
#  - Newton_m(dJ1,dJ2, 1.0) # Significantly faster. Also slightly better CV?
# => Despite inconvienience of defining analytic derivatives,
#    Newton_m seems like the best option.
#  - In extreme (or just non-linear Obs.mod) cases,
#    the EnKF-N cost function may have multiple minima.
#    Then: should use more robust optimizer!
#
# For 'primal'
# ----------------------------------------
# Similarly, Newton_m seems like the best option,
# although alternatives are provided (commented out).
#
def Newton_m(fun, deriv, x0, is_inverted=False,
             conf=1.0, xtol=1e-4, ytol=1e-7, itermax=10**2):
    """Find root of `fun`.

    This is a simple (and pretty fast) implementation of Newton's method.
    """
    itr = 0
    dx = np.inf
    Jx = fun(x0)

    def norm(x):
        return sqrt(np.sum(x**2))
    while ytol < norm(Jx) and xtol < norm(dx) and itr < itermax:
        Dx  = deriv(x0)
        if is_inverted:
            dx  = Dx @ Jx
        elif isinstance(Dx, float):
            dx  = Jx/Dx
        else:
            dx  = mldiv(Dx, Jx)
        dx *= conf
        x0 -= dx
        Jx  = fun(x0)
        itr += 1
    return x0


def hyperprior_coeffs(s, N, xN=1, g=0):
    r"""Set EnKF-N inflation hyperparams.

    The EnKF-N prior may be specified by the constants:

    - `eN`: Effect of unknown mean
    - `cL`: Coeff in front of log term

    These are trivial constants in the original EnKF-N,
    but are further adjusted (corrected and tuned) for the following reasons.

    - Reason 1: mode correction.
      These parameters bridge the Jeffreys (`xN=1`) and Dirac (`xN=Inf`) hyperpriors
      for the prior covariance, B, as discussed in `bib.bocquet2015expanding`.
      Indeed, mode correction becomes necessary when $$ R \rightarrow \infty $$
      because then there should be no ensemble update (and also no inflation!).
      More specifically, the mode of `l1`'s should be adjusted towards 1
      as a function of $$ I - K H $$ ("prior's weight").
      PS: why do we leave the prior mode below 1 at all?
      Because it sets up "tension" (negative feedback) in the inflation cycle:
      the prior pulls downwards, while the likelihood tends to pull upwards.

    - Reason 2: Boosting the inflation prior's certainty from N to xN*N.
      The aim is to take advantage of the fact that the ensemble may not
      have quite as much sampling error as a fully stochastic sample,
      as illustrated in section 2.1 of `bib.raanes2019adaptive`.

    - Its damping effect is similar to work done by J. Anderson.

    The tuning is controlled by:

    - `xN=1`: is fully agnostic, i.e. assumes the ensemble is generated
      from a highly chaotic or stochastic model.
    - `xN>1`: increases the certainty of the hyper-prior,
      which is appropriate for more linear and deterministic systems.
    - `xN<1`: yields a more (than 'fully') agnostic hyper-prior,
      as if N were smaller than it truly is.
    - `xN<=0` is not meaningful.
    """
    N1 = N-1

    eN = (N+1)/N
    cL = (N+g)/N1

    # Mode correction (almost) as in eqn 36 of `bib.bocquet2015expanding`
    prior_mode = eN/cL                        # Mode of l1 (before correction)
    diagonal   = pad0(s**2, N) + N1           # diag of Y@R.inv@Y + N1*I
    #                                           (Hessian of J)
    I_KH       = np.mean(diagonal**(-1))*N1   # ≈ 1/(1 + HBH/R)
    # I_KH      = 1/(1 + (s**2).sum()/N1)     # Scalar alternative: use tr(HBH/R).
    mc         = sqrt(prior_mode**I_KH)       # Correction coeff

    # Apply correction
    eN /= mc
    cL *= mc

    # Boost by xN
    eN *= xN
    cL *= xN

    return eN, cL


def zeta_a(eN, cL, w):
    """EnKF-N inflation estimation via w.

    Returns `zeta_a = (N-1)/pre-inflation^2`.

    Using this inside an iterative minimization as in the
    `dapper.da_methods.variational.iEnKS` effectively blends
    the distinction between the primal and dual EnKF-N.
    """
    N  = len(w)
    N1 = N-1
    za = N1*cL/(eN + w@w)
    return za


@ens_method
class EnKF_N:
    """Finite-size EnKF (EnKF-N).

    Refs: `bib.bocquet2011ensemble`, `bib.bocquet2015expanding`

    This implementation is pedagogical, prioritizing the "dual" form.
    In consequence, the efficiency of the "primal" form suffers a bit.
    The primal form is included for completeness and to demonstrate equivalence.
    In `dapper.da_methods.variational.iEnKS`, however,
    the primal form is preferred because it
    already does optimization for w (as treatment for nonlinear models).

    `infl` should be unnecessary (assuming no model error, or that Q is correct).

    `Hess`: use non-approx Hessian for ensemble transform matrix?

    `g` is the nullity of A (state anomalies's), ie. g=max(1,N-Nx),
    compensating for the redundancy in the space of w.
    But we have made it an input argument instead, with default 0,
    because mode-finding (of p(x) via the dual) completely ignores this redundancy,
    and the mode gets (undesireably) modified by g.

    `xN` allows tuning the hyper-prior for the inflation.
    Usually, I just try setting it to 1 (default), or 2.
    Further description in hyperprior_coeffs().
    """

    N: int
    dual: bool = False
    Hess: bool = False
    xN: float  = 1.0
    g: int     = 0

    def assimilate(self, HMM, xx, yy):
        N, N1 = self.N, self.N-1

        # Init
        E = HMM.X0.sample(N)
        self.stats.assess(0, E=E)

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # Forecast
            E = HMM.Dyn(E, t-dt, dt)
            E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)

            # Analysis
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)
                Eo = HMM.Obs(ko)(E)
                y  = yy[ko]

                mu = np.mean(E, 0)
                A  = E - mu

                xo = np.mean(Eo, 0)
                Y  = Eo-xo
                dy = y - xo

                R = HMM.Obs(ko).noise.C
                V, s, UT = svd0(Y @ R.sym_sqrt_inv.T)
                du       = UT @ (dy @ R.sym_sqrt_inv.T)
                def dgn_N(l1): return pad0((l1*s)**2, N) + N1

                # Adjust hyper-prior
                # xN_ = noise_level(self.xN, self.stats, HMM.tseq, N1, ko, A,
                #                   locals().get('A_old', None))
                eN, cL = hyperprior_coeffs(s, N, self.xN, self.g)

                if self.dual:
                    # Make dual cost function (in terms of l1)
                    def pad_rk(arr): return pad0(arr, min(N, len(y)))
                    def dgn_rk(l1): return pad_rk((l1*s)**2) + N1

                    def J(l1):
                        val = np.sum(du**2/dgn_rk(l1)) \
                            + eN/l1**2 \
                            + cL*np.log(l1**2)
                        return val

                    # Derivatives (not required with minimize_scalar):
                    def Jp(l1):
                        val = -2*l1 * np.sum(pad_rk(s**2) * du**2/dgn_rk(l1)**2) \
                            + -2*eN/l1**3 + 2*cL/l1
                        return val

                    def Jpp(l1):
                        val = 8*l1**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l1)**3) \
                            + 6*eN/l1**4 + -2*cL/l1**2
                        return val
                    # Find inflation factor (optimize)
                    l1 = Newton_m(Jp, Jpp, 1.0)
                    # l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
                    # l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2),
                    #                      tol=1e-4).x

                else:
                    # Primal form, in a fully linearized version.
                    def za(w): return zeta_a(eN, cL, w)

                    def J(w): return \
                        .5*np.sum(((dy-w@Y)@R.sym_sqrt_inv.T)**2) + \
                        .5*N1*cL*np.log(eN + w@w)
                    # Derivatives (not required with fmin_bfgs):
                    def Jp(w): return -Y@R.inv@(dy-w@Y) + w*za(w)
                    # Jpp   = lambda w:  Y@R.inv@Y.T + \
                    #     za(w)*(eye(N) - 2*np.outer(w,w)/(eN + w@w))
                    # Approx: no radial-angular cross-deriv:
                    # Jpp   = lambda w:  Y@R.inv@Y.T + za(w)*eye(N)

                    def nvrs(w):
                        # inverse of Jpp-approx
                        return (V * (pad0(s**2, N) + za(w)) ** -1.0) @ V.T
                    # Find w (optimize)
                    wa     = Newton_m(Jp, nvrs, zeros(N), is_inverted=True)
                    # wa   = Newton_m(Jp,Jpp ,zeros(N))
                    # wa   = fmin_bfgs(J,zeros(N),Jp,disp=0)
                    l1     = sqrt(N1/za(wa))

                # Uncomment to revert to ETKF
                # l1 = 1.0

                # Explicitly inflate prior
                # => formulae look different from `bib.bocquet2015expanding`.
                A *= l1
                Y *= l1

                # Compute sqrt update
                Pw = (V * dgn_N(l1)**(-1.0)) @ V.T
                w  = dy@R.inv@Y.T@Pw
                # For the anomalies:
                if not self.Hess:
                    # Regular ETKF (i.e. sym sqrt) update (with inflation)
                    T = (V * dgn_N(l1)**(-0.5)) @ V.T * sqrt(N1)
                    # = (Y@R.inv@Y.T/N1 + eye(N))**(-0.5)
                else:
                    # Also include angular-radial co-dependence.
                    # Note: denominator not squared coz
                    # unlike `bib.bocquet2015expanding` we have inflated Y.
                    Hw = Y@R.inv@Y.T/N1 + eye(N) - 2*np.outer(w, w)/(eN + w@w)
                    T  = funm_psd(Hw, lambda x: x**-.5)  # is there a sqrtm Woodbury?

                E = mu + w@A + T@A
                E = post_process(E, self.infl, self.rot)

                self.stats.infl[ko] = l1
                self.stats.trHK[ko] = (((l1*s)**2 + N1)**(-1.0)*s**2).sum()/len(y)

            self.stats.assess(k, ko, E=E)
