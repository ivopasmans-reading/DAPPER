"""Time sequence management, notably Chronology and Ticker."""

import colorama
import numpy as np
from struct_tools import AlignedDict

from dapper.tools.colors import color_text


def dko_iterator(self, kko):
    for ko0, ko1 in zip([0]+kko[:-1], kko):
        yield ko1-ko0
        
def smod(counter, denominator, *args):
    prem =  np.mod( counter, denominator, *args)
    mrem = -np.mod(-counter, denominator, *args)
    return prem if np.abs(prem)<=np.abs(mrem) else mrem

class Chronology():
    """Time schedules with consistency checks.

    - Uses int records, so `tt[k] == k*dt`.
    - Uses generators, so time series may be arbitrarily long.

    Example illustration:

                             [----dto------]
                      [--dt--]
        tt:    0.0    0.2    0.4    0.6    0.8    1.0    T
        kk:    0      1      2      3      4      5      K
               |------|------|------|------|------|------|
        ko:    None   None   0      None   1      None   Ko
        kko:                 2             4             6
                             [----dko------]

    .. warning:: By convention, there is no obs at 0.
                 This is hardcorded in DAPPER,
                 whose cycling starts by the forecast.

    Identities (subject to precision):

        len(kk)  == len(tt)  == K   +1
        len(kko) == len(tto) == Ko+1

        kko[0]   == dko      == dto/dt == K/(Ko+1)
        kko[-1]  == K        == T/dt
        Ko       == T/dto-1

    These attributes may be set (altered) after init: `dt, dko, K, T`.
    Setting other attributes (alone) is ambiguous
    (e.g. should `dto*=2` yield a doubling of `T` too?),
    and so should/will raise an exception.
    """

    def __init__(self, dt=None, dto=None, T=None, BurnIn=-1,
                 dko=None, Ko=None, K=None, Tplot=None,
                 kko=None, kk=None):
        
        #Define time step. 
        if dt is not None:
            self._dt = dt
        elif T and K:
            self._dt = T/K 
        elif dko and dto:
            self._dt = dto/dko 
        elif T and dko and Ko:
            self._dt = T/(Ko+1)/dko 
        elif T and kko:
            self._dt = T/self.kko[-1]
        else:
            raise TypeError('Unable to derive dt.')
        
        #Define kk 
        if K is not None:
            self._K = K 
        elif kk:
            self._K = int(kk[-1])
        elif T:
            self._K = int(T / self.dt)
        elif Ko and dto:
            self._K = (Ko+1) * dto / dt
        else:
            raise TypeError('Unable to derive K.')
        self.kk = np.arange(0, self.K+1, dtype=int)
        
        #Define kko 
        if kko is not None:
            self._kko = kko 
        elif dko:
            self._kko = self.kk[dko::dko]
        elif dto:
            dko = int(dto/dt)
            self._kko = self.kk[dko::dko]
        else:
            raise TypeError('Unable to derive kko.')
        
        if Ko:
            self.kko = self.kko[:Ko+1]

        # BurnIn, Tplot
        if BurnIn is None:
            self.BurnIn = 0.0
        elif self.T <= BurnIn:
            self.BurnIn = self.T / 2
            warning = "Warning: experiment duration < BurnIn time." \
                      "\nReducing BurnIn value."
            print(color_text(warning, colorama.Fore.RED))
        else:
            self.BurnIn = BurnIn
            
        if Tplot is None:
            self.Tplot = BurnIn
        else:
            self.Tplot = Tplot
            
        self.times = np.arange(0, self.K+1) * self.dt 
        if len(self.kko)==0:
            self.otimes = np.array([])
        else:
            self.otimes = self.times[self.kko]

    ######################################
    # "State vars". Can be set (changed).
    ######################################

    # @dt.setter
    # def dt(self, value):
    #     dko_new = self.dko * self.dt/value
    #     if not np.isclose(int(dko_new), dko_new):
    #         raise ValueError('New value is amgiguous with respect to dko')
    #     dko_new = int(dko_new)
    #     self.__init__(dt=value, K=K, kko=kko, dko=dko, BurnIn=self.BurnIn, Tplot=self.Tplot)

    # @property
    # def dko(self):
    #     return self._dko
    #
    # @dko.setter
    # def dko(self, value):
    #     ratio = value/self.dko
    #     self.__init__(dt=self.dt, dko=value, T=ratio*self.T,
    #                   BurnIn=self.BurnIn, Tplot=self.Tplot)

    # @property
    # def K(self):
    #     return self._K

    # @K.setter
    # def K(self, value):
    #     self.__init__(dt=self.dt, dko=self.dko, K=value,
    #                   BurnIn=self.BurnIn, Tplot=self.Tplot)

    ######################################
    # Read/write (though not state var)
    ######################################
    # @property
    # def T(self):
    #     return self.dt*self.K

    # @T.setter
    # def T(self, value):
    #     self.__init__(dt=self.dt, dko=self.dko, T=value,
    #                   BurnIn=self.BurnIn, Tplot=self.Tplot)

    # @property
    # def Ko(self):
    #     return len(self.kko)-1
    #
    # @Ko.setter
    # def Ko(self, value):
    #     self.__init__(dt=self.dt, dko=self.dko, Ko=value,
    #                   BurnIn=self.BurnIn, Tplot=self.Tplot)

    ######################################
    # Read-only
    ######################################
    @property
    def dto(self):
        return self.dko * self.dt
        
    @property
    def dko(self):
        if len(self.kko)==0:
            return np.nan
        if np.all( np.mod(self.kko, self.kko[0])==0 ):
            return self.kko[0]
        else:
            raise ValueError("Nonuniform observation timestep.")
    
    @property 
    def dt(self):
        return self._dt 
    
    @property
    def K(self):
        return self._K
    
    @property 
    def kko(self):
        return self._kko
    
    @property 
    def tt(self):
        return self.kk * self.dt 
    
    @property 
    def tto(self):
        return self.kko * self.dt 
    
    @property
    def Ko(self):
        return len(self.kko)-1
    
    @property
    def T(self):
        return self.K * self.dt

    # Burn In. NB: uses > (strict inequality)
    @property
    def mask(self):
        """Example use: `kk_BI = kk[mask]`"""
        return self.tt > self.BurnIn

    @property
    def masko(self):
        """Example use: `kko_BI = kko[masko]`"""
        return self.tto > self.BurnIn

    @property
    def iBurnIn(self):
        return self.mask.nonzero()[0][0]

    @property
    def ioBurnIn(self):
        return self.masko.nonzero()[0][0]

    ######################################
    # Other
    ######################################
    @property
    def ticker(self):
        """Fancy version of `range(1,K+1)`.

        Also yields `t`, `dt`, and `ko`.
        """
        tckr = Ticker(self.tt, self.kko)
        next(tckr)
        return tckr

    def cycle(self, ko):
        """The range (in `kk`) between observation `ko-1` and `ko`.

        Also yields `t` and `dt`.
        """
        for k in np.arange(0 if ko==0 else self.kko[ko-1], self.kko[ko])+1:
            t  = self.tt[k]
            dt = t - self.tt[k-1]
            yield k, t, dt

    def __str__(self):
        printable = ['K', 'Ko', 'T', 'BurnIn', 'dto', 'dt']
        return str(AlignedDict([(k, getattr(self, k)) for k in printable]))

    def __repr__(self):
        return "<" + type(self).__name__ + '>' + "\n" + str(self)

    ######################################
    # Utilities
    ######################################
    def copy(self):
        """Copy via state vars."""
        return Chronology(dt=self.dt, K=self.K, kko=self.kko, BurnIn=self.BurnIn, Tplot=self.Tplot)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Ticker:
    """Iterator over kk and `kko`, yielding `(k,ko,t,dt)`.

    Includes `__len__` for progressbar usage.

    `ko = kko.index(k)`, or `None` otherwise,
    but computed without this repeated look-up operation.
    """

    def __init__(self, tt, kko):
        self.tt  = tt
        self.kko = kko
        self.reset()

    def reset(self):
        self.k   = 0
        self._ko = 0
        self.ko  = None

    def __len__(self):
        return len(self.tt) - self.k

    def __iter__(self): return self

    def __next__(self):
        if self.k >= len(self.tt):
            raise StopIteration
        t    = self.tt[self.k]
        dt   = t - self.tt[self.k-1] if self.k > 0 else np.NaN
        item = (self.k, self.ko, t, dt)
        self.k += 1
        if self._ko < len(self.kko) and self.k == self.kko[self._ko]:
            self.ko = self._ko
            self._ko += 1
        else:
            self.ko = None
        return item


def format_time(k, ko, t):
    if k is None:
        k    = "init"
        t    = "init"
        ko = "N/A"
    else:
        t    = "   t=%g" % t
        k    = "   k=%d" % k
        ko = "ko=%s" % ko
    s = "\n".join([t, k, ko])
    return s
