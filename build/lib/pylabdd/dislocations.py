# Module pylabdd.dislocations
'''Module pylabdd.dislocations introduces class ``Dislocations`` that contains attributes 
and methods needed to handle a dislocation configuration. 

uses NumPy and MatPlotLib.pyplot

Version: 1.2 (2024-01-10)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, December 2023
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''

import logging
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

try:
    WORK_DIR = os.environ['CONDA_PREFIX']  # if path to conda env is set, use this one
    path = os.path.join(WORK_DIR, 'PATHS.txt')
    with open(path, 'r') as f:
        MAIN_DIR = f.read()  # directory in which repository is cloned
except Exception as e:
    # otherwise fall back to user home
    WORK_DIR = os.path.join(os.path.expanduser('~'), '.pylabdd')
    path = os.path.join(WORK_DIR, 'PATHS.txt')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            MAIN_DIR = f.read()  # directory in which repository is cloned
    else:
        logging.error(f'No path information for fortran module in conda env or user home: {e}')
        logging.error('Trying CWD.')
        MAIN_DIR = os.getcwd()
        if MAIN_DIR[-9:] == 'notebooks':
            MAIN_DIR = MAIN_DIR[:-10]  # remove '/notebooks' from end of path
    
try:
    CWD = os.getcwd()
    os.chdir(MAIN_DIR)
    from PK_force import calc_fpk_pbc, calc_fpk
    os.chdir(CWD)
    print('Using Fortran subroutine for PK force.')
except Exception as e:
    logging.error(f'An exception has occurred while importing fortran module: {e}')
    from pylabdd.pkforce import calc_fpk_pbc, calc_fpk
    print('Using Python subroutines for PK force.')


#define class for dislocations
class Dislocations:
    '''Define class for Dislocations

    Parameters
    ----------
    Nm : int
        Number of mobile dislocations
    Ns : int
        Number of sources

    Attributes
    ----------
    spi1 : Nd-array
        Slip plane inclination angles
    mu : float
        Shear modulus
    nu : float
        Poisson's ratio
    b0 : float
        Norm of Burgers vector
    dmob : float
        Dislocation mobility
    f0 : float
        Lattice friction stress
    m : int
        Stress exponent
    dmax : float
        Max. distance a dislocation can move in one time step
    xpos : Nd-array
        x-positions of dislocations
    ypos : Nd-array
        y-positions of dislocations
    LX : float
        x-dimension of domain
    LY : float
        y-dimension of domain
    bc : str
        Boundary condition
    dt0 : float
        Current time-step size
    flag : Nd-array
        Flags for dislocation types (0: mobile; 1: source)
    ctau :
        Frank-Read source critical nucleation stress
    tnuc :
        Frank-Read source critical nucleation time
    dtmax :
        Max time-step
    dtmin :
        Min time-step
    cutoff :
        Cutoff for stress evaluation (if 0, stresses will be infinite at core)
    postdt : boolean
        Choose whether to engage force and displacement correcting (mat be broken)

    '''
    def __init__(self, Nm, Ns, mu, nu, b0, spi1=0., \
                dmob=1., f0=0.8, m=7, dmax=0.02, \
                xpos=None, ypos=None,\
                LX=10., LY=10., bc='pbc',\
                dt0=0.02, \
                flag=None, \
                ctau=110., tnuc=10.e-3, \
                dtmax=5.e-3, dtmin=0.5e-3, cutoff=1e-2, postdt=True, \
                Lnuc=None
                ):
        self.Nmob = int(Nm)      # number of mobile dislocations
        self.Nsrc = int(Ns)      # number of sources
        Nd = Nm + Ns # total number of dislocations
        self.Ntot = Nd

        #dislocation positions
        if xpos is None:
            self.xpos = np.zeros(Nd)
        else:
            if 'float' in str(type(xpos)):
                xpos = [xpos]
            self.xpos = np.array(xpos)
        if ypos is None:
            self.ypos = np.zeros(Nd)
        else:
            if 'float' in str(type(ypos)):
                ypos = [ypos]
            self.ypos = np.array(ypos)
        self.xpos = self.xpos.astype(float)
        self.ypos = self.ypos.astype(float)
        self.dx   = np.zeros(Nd)
        self.dy   = np.zeros(Nd)
        self.xpeq = None # equilibrium positions, will be defined in Dislocation.relax_disl
        self.ypeq = None

        # Dislocation flags
        if flag is None:
            self.flag = np.zeros(Nd).astype(int)
            self.flag[Nm:] = 1
        else:
            self.flag = np.array(flag).astype(int)

        #self.Nimm = sum(self.flag == 7) # Number of immobile dislocations

        #slip plane inclination angles
        self.sp_inc = np.ones(Nd)*spi1
        self.sp_inc = self.sp_inc.astype(float)

        #Burgers vectors
        self.bx = np.cos(self.sp_inc)
        self.by = np.sin(self.sp_inc)

        #calculate dislocation densities
        self.rho = Nd/(LX*LY)
        self.rho_m = Nm/(LX*LY)

        #dislocation mobility parameters
        self.mu   = float(mu)
        self.nu   = float(nu)
        self.b0   = float(b0)     # norm of Burgers vector
        self.C    = self.mu*self.b0 / (2*np.pi*(1.-self.nu))      # elastic parameter C=mu*b0/(2*pi*(1.-nu))
        self.dmob = float(dmob)   # dislocation mobility
        self.f0   = float(f0)     # lattice friction stress
        self.m    = m      # stress exponent
        self.dmax = float(dmax)   # max. distance a dislocation can move in one time step

        # Frank-Read sources
        #self.Nsrc = sum(self.flag == 1) # Number of sources
        self.dtnuc0 = np.zeros(self.Ntot) # Source nucleation timer
        #self.ctau = ctau * np.ones(self.Ntot) # Source critical nucleation stress
        self.ctau = float(ctau)
        #self.tnuc = tnuc * np.ones(self.Ntot) # Source critical nucleation time
        self.tnuc = tnuc
        if Lnuc is None:
          self.Lnuc = self.C / self.ctau
        else:
          self.Lnuc = Lnuc

        #geometry of the domain
        self.lx = float(LX)      # x-dimension of domain
        self.ly = float(LY)      # y-dimension of domain
        self.bc = bc      # chose between 'pbc' and 'fixed'
        if bc!='pbc' and bc!='fixed' and bc!='finite' and bc!='finiteFEM':
            raise ValueError('BC not defined: '+bc)

        #numerical parameters
        self.dt0 = float(dt0)
        self.dtmax = float(dtmax)
        self.dtmin = float(dtmin)
        self.cutoff = float(cutoff)
        self.postdt = postdt

    #define functions for stress field evaluation
    def sig_xx(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return -self.C*Y*(3.*hx + hy)/(hh*hh + self.cutoff)

    def sig_yy(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return self.C*Y*(hx - hy)/(hh*hh + self.cutoff)

    def sig_xy(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return self.C*X*(hx - hy)/(hh*hh + self.cutoff)

    #create multiple new dislocations
    def create_dislocations(self, x, y, sp_inc, dx, dy, flag):
        assert x.size == y.size == sp_inc.size == dx.size == dy.size == flag.size
        Nc = x.size
        self.xpos = np.concatenate((x,self.xpos),axis=0)
        self.ypos = np.concatenate((y,self.ypos),axis=0)
        self.sp_inc = np.concatenate((sp_inc,self.sp_inc),axis=0)
        bx = np.cos(sp_inc)
        by = np.sin(sp_inc)
        self.bx = np.concatenate((bx,self.bx),axis=0)
        self.by = np.concatenate((by,self.by),axis=0)
        self.flag = np.concatenate((flag,self.flag), axis=0)
        self.dx = np.concatenate((dx,self.dx),axis=0)
        self.dy = np.concatenate((dy,self.dy),axis=0)
        self.Ntot += Nc
        self.Nmob += Nc
        self.rho = self.Ntot/(self.lx*self.ly)
        self.rho_m = self.Nmob/(self.lx*self.ly)
        self.dtnuc0 = np.concatenate((np.zeros(Nc),self.dtnuc0),axis=0)

    def update_source_nucleation(self, tau0, dt, sxx=None, syy=None, sxy=None):
        """
        Calculates resolved shear stress on sources,
        updates their nucleation timer,
        and creates new mobile dislocations when nucleations conditions are met.
        NOTE: sources generate a stress field is if they were mobile dislocations
        """
        isrc = np.logical_and.reduce([self.flag == 1])
        if sum(isrc) <= 0:
            return
        # Calculate slip plane resolved shear stress for FR sources
        rstau = np.zeros(self.Ntot)
        # Due to some global uniform stress
        rstau[isrc] = tau0
        # Due to other dislocations
        rstau[isrc] += self.calc_select_rstau( \
            self.xpos[isrc], self.ypos[isrc], self.bx[isrc], self.by[isrc])
        # Due to external hat stress
        if sxx is not None and syy is not None and sxy is not None:
            rstau[isrc] += self.calc_rstau_external( \
                self.bx[isrc], self.by[isrc], sxx[isrc], syy[isrc], sxy[isrc])

        # Select sources with stress above nucleation stress:
        ih = np.logical_and.reduce([self.flag == 1, abs(rstau) > self.ctau])
        self.dtnuc0[ih] += dt
        # Select sources with stress above nucleation stress for longer than nucleation time:
        inuc = np.logical_and.reduce([ih, self.dtnuc0 > self.tnuc])
        if sum(inuc) <= 0:
            return
        self.dtnuc0[inuc] = 0 # Reset nucleation timer

        # Anti-symmetric and symmetric dislocation creation
        symf = np.sign(rstau[inuc])
        flag = np.zeros(sum(inuc))

        sp_inc1 = self.sp_inc[inuc]
        dx1 = symf * self.Lnuc * np.cos(sp_inc1)
        dy1 = symf * self.Lnuc * np.sin(sp_inc1)
        x1 = self.xpos[inuc] + dx1
        y1 = self.ypos[inuc] + dy1

        sp_inc2 = self.sp_inc[inuc] + np.pi
        dx2 = symf * self.Lnuc * np.cos(sp_inc2)
        dy2 = symf * self.Lnuc * np.sin(sp_inc2)
        x2 = self.xpos[inuc] + dx2
        y2 = self.ypos[inuc] + dy2

        self.create_dislocations(x1, y1, sp_inc1, dx1, dy1, flag)
        self.create_dislocations(x2, y2, sp_inc2, dx2, dy2, flag)

    def calc_select_rstau(self, xp, yp, bx, by):
        sxx = np.zeros(xp.size)
        syy = np.zeros(xp.size)
        sxy = np.zeros(xp.size)
        for i in range(self.Nmob):
            sxx += self.bx[i]*self.sig_xx(xp-self.xpos[i],yp-self.ypos[i])
            sxx += self.by[i]*self.sig_yy(xp-self.xpos[i],yp-self.ypos[i])
            syy += self.bx[i]*self.sig_yy(xp-self.xpos[i],yp-self.ypos[i])
            syy += self.by[i]*self.sig_xx(xp-self.xpos[i],yp-self.ypos[i])
            sxy += self.bx[i]*self.sig_xy(xp-self.xpos[i],yp-self.ypos[i])
            sxy += self.by[i]*self.sig_xy(xp-self.xpos[i],yp-self.ypos[i])
            # sxx = self.sig_xx(xp-self.xpos[i],yp-self.ypos[i])
            # syy = self.sig_yy(xp-self.xpos[i],yp-self.ypos[i])
            # sxy = self.sig_xy(xp-self.xpos[i],yp-self.ypos[i])
        taux = sxx*bx + sxy*by
        tauy = sxy*bx + syy*by
        rstau = tauy*bx - taux*by
        return rstau

    def calc_select_fsp(self, ih=None, Nd=None, xp=None, yp=None, Nm=None, tau0=None,
                   lx=None, ly=None, bx=None, by=None):
        if Nd is None:
            Nd = self.Ntot
        if xp is None:
            xp = self.xpos
        if yp is None:
            yp = self.ypos
        if Nm is None:
            Nm = self.Nmob
        if tau0 is None:
            tau0 = 0.
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly
        if bx is None:
            bx = self.bx
        if by is None:
            by = self.by
        if ih is not None and sum(ih) != 0:
            Ncalc = sum(ih)
            xp = np.concatenate((xp[ih],np.delete(xp, ih, axis=0)),axis=0)
            yp = np.concatenate((xp[ih],np.delete(xp, ih, axis=0)),axis=0)
            bx = np.concatenate((bx[ih],np.delete(bx, ih, axis=0)),axis=0)
            by = np.concatenate((by[ih],np.delete(by, ih, axis=0)),axis=0)
        elif sum(ih) == 0:
            fsp = []
            return fsp
        else:
            Ncalc = self.Nmob
        FPK = np.zeros((2,Ncalc))
        if self.bc=='pbc':
            if Nd > 0:
                FPK = self.C*calc_fpk_pbc(xp[0:Nd], yp[0:Nd], bx[0:Nd], by[0:Nd], tau0,
                                          lx, ly, Ncalc, Nd)
        else:
            if Nd > 0:
                FPK = self.C*calc_fpk(xp[0:Nd], yp[0:Nd], bx[0:Nd], by[0:Nd], tau0, Ncalc, Nd)
        fsp = np.sum(np.multiply(FPK,np.abs(np.array([bx[0:Ncalc],by[0:Ncalc]]))), axis=0)
        return fsp

    def calc_rstau_external(self, bx, by, sxx, syy, sxy):
        taux = sxx*bx + sxy*by
        tauy = sxy*bx + syy*by
        rstau = tauy*bx - taux*by
        return rstau

    def calc_fpk_external(self, bx, by, sxx, syy, sxy):
        assert bx.size == by.size == sxx.size == syy.size == sxy.size
        N = bx.size
        FPK = np.zeros((2, N), dtype=np.float64)
        tx = sxx*bx + sxy*by
        ty = sxy*bx + syy*by
        FPK = np.vstack((ty,-tx))
        return FPK

    def calc_force(self, xp=None, yp=None, Nm=None, tau0=None,
                   lx=None, ly=None):
        if xp is None:
            xp = self.xpos
        if yp is None:
            yp = self.ypos
        if Nm is None:
            Nm = self.Nmob
        if tau0 is None:
            tau0 = 0.
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly
        FPK = np.zeros((2,Nm))
        if self.bc=='pbc':
            if self.Ntot > 0:
                FPK = self.C*calc_fpk_pbc(xp[0:self.Ntot], yp[0:self.Ntot], self.bx[0:self.Ntot], self.by[0:self.Ntot], tau0,
                                          lx, ly, Nm, self.Ntot)
        else:
            if self.Ntot > 0:
                FPK = self.C*calc_fpk(xp[0:self.Ntot], yp[0:self.Ntot], self.bx[0:self.Ntot], self.by[0:self.Ntot], tau0, Nm, self.Ntot)
        return FPK

    #initialize random dislocation positions
    def positions(self, stol=0.25):
        #select slip planes first by random sequential algorithm
        #make sure that slip planes are at least a distance of stol apart
        self.ypos[0] = self.ly*np.random.rand(1)[0]
        isl = 1
        while isl<self.Ntot:
            hy = self.ly*np.random.rand(1)[0]
            flag = np.logical_and(self.ypos[0:isl]<hy+stol, self.ypos[0:isl]>hy-stol)
            if not np.any(flag):
                self.ypos[isl] = hy
                isl += 1

        #place dislocations randomnly on slip planes
        hh = np.random.rand(self.Ntot)
        self.xpos = self.lx*hh
        self.ypos += np.sin(self.sp_inc)*hh*self.ly
        ih = np.nonzero(self.ypos<0.)[0]
        self.ypos[ih] += self.ly
        ih = np.nonzero(self.ypos>self.ly)[0]
        self.ypos[ih] -= self.ly

        #bx = np.multiply(bx, np.sign(np.random.rand(N)-0.5))  # random positive and negative Burgers vectors
        self.bx[0:self.Ntot:2] *= -1. # change sign of every second dislocation
        self.by[0:self.Ntot:2] *= -1.

    #define force norm for relaxation with L-BFGS-B method
    def fnorm(self, dr, tau0, Nm):
        xp = self.xpos
        yp = self.ypos
        dx = np.multiply(dr, np.abs(self.bx[0:Nm]))
        dy = np.multiply(dr, np.abs(self.by[0:Nm]))
        xp[0:Nm] += dx
        yp[0:Nm] += dy
        FPK = np.zeros((2,Nm))
        if self.bc=='pbc':
            if self.Ntot > 0:
                FPK = self.C*calc_fpk_pbc(xp[0:self.Ntot], yp[0:self.Ntot], self.bx[0:self.Ntot], self.by[0:self.Ntot], tau0,
                                              self.lx, self.ly, Nm, self.Ntot)
        else:
            if self.Ntot > 0:
                FPK = self.C*calc_fpk(xp[0:self.Ntot], yp[0:self.Ntot], self.bx[0:self.Ntot], self.by[0:self.Ntot], tau0, Nm, self.Ntot)
        fsp = np.sum(np.multiply(FPK,np.absolute(np.array([self.bx[0:Nm],\
                                                self.by[0:Nm]]))), axis=0)
        return np.sum(np.absolute(fsp))/Nm

    #calculate dislocation velocity
    def dvel(self, fsp, ml):
        if ml=='viscous':
            hh = fsp
        elif ml=='powerlaw':
            hh = np.multiply(np.abs(fsp/self.f0)**self.m, np.sign(fsp))
        else:
            raise ValueError('Dislocation mobility ""'+ml+'" not supported.')
        return hh*self.dmob

    #update dislocation positions
    def move_disl(self, tau0, ml, dt, bc=None, sigextxx=None, sigextyy=None, sigextxy=None):
        Nm = self.Nmob
        Ns = self.Nsrc
        Ncalc = self.Ntot-Ns
        FPK = np.zeros((2,Nm))
        if bc is None:
            bc = self.bc
        if bc=='pbc':
            if Ncalc > 0:
                FPK = self.C*calc_fpk_pbc(self.xpos[0:Ncalc], self.ypos[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], \
                                            tau0, self.lx, self.ly, Nm, Ncalc)
            FPK[:][1] *= -1.
            #define maximum dislocation displacement
            lb = -self.dmax
            ub = self.dmax
        elif bc=='fixed':
            if Ncalc > 0:
                FPK = self.C*calc_fpk(self.xpos[0:Ncalc], self.ypos[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
            #define possible range to move a dislocation within box
            lb = -np.minimum(np.abs(self.xpos[0:Nm]/self.bx[0:Nm]), np.ones(Nm)*self.dmax)
            ub =  np.minimum(np.abs((self.lx-self.xpos[0:Nm])/self.bx[0:Nm]), np.ones(Nm)*self.dmax)
        elif bc=='finite':
            # DHR: Dislocations disappear after exiting box
            #tau0 = 0
            if Ncalc > 0:
                FPK = self.C*calc_fpk(self.xpos[0:Ncalc], self.ypos[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
            lb = -self.dmax
            ub = self.dmax
        elif bc=='finiteFEM':
            if Ncalc > 0:
                FPK = self.C*calc_fpk(self.xpos[0:Ncalc], self.ypos[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
            if Nm > 0:
                FPK += self.calc_fpk_external(self.bx[0:Nm],self.by[0:Nm],\
                                            sigextxx[0:Nm],sigextyy[0:Nm],sigextxy[0:Nm])
            lb = -self.dmax
            ub = self.dmax
        else:
            raise ValueError('BC not defined: '+bc)
        # fsp = np.sum(np.multiply(FPK,np.abs(np.array([self.bx[0:Nm],self.by[0:Nm]]))), axis=0)
        fsp = np.sum(np.multiply(FPK,np.array([self.bx[0:Nm],self.by[0:Nm]])), axis=0)
        fsp[np.isnan(fsp)] = self.cutoff
        drp = self.dvel(fsp, ml)*dt  # predictor for simple forward Euler integration dr = v.dt
        drp = np.clip(drp, lb, ub)  # enforce speed limit for dislocations and make sure they stay in box
        dr  = np.zeros(self.Ntot)
        dr[0:Nm] += drp  # only Nm dislocations are moved, the rest is fixed
        #do some analysis for time step control
        hh = np.abs(dr)
        dr_max = np.amax(hh)
        nmax = np.nonzero(hh>=self.dmax)[0]
        self.dx = np.multiply(dr, self.bx)  # projection on slip plane (defined by B-vector)
        self.dy = np.multiply(dr, self.by)
        xp = self.xpos + self.dx
        yp = self.ypos + self.dy
        #verify if force after predictor step has same sign as before
        #if not dislocation passes a minimum position and needs a reduced time step
        ih = np.array([1, 1])  # initialize ih such that while is performed at least once
        jc = 0
        while len(ih)>0 and jc<5 and self.postdt:
            if bc=='pbc':
                if Ncalc > 0:
                    FPK = self.C*calc_fpk_pbc(xp[0:Ncalc], yp[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], \
                                                tau0, self.lx, self.ly, Nm, Ncalc)
            elif bc=='fixed':
                if Ncalc > 0:
                    FPK = self.C*calc_fpk(xp[0:Ncalc], yp[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
            elif bc=='finite':
                if Ncalc > 0:
                    FPK = self.C*calc_fpk(xp[0:Ncalc], yp[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
            elif bc=='finiteFEM':
                if Ncalc > 0:
                    FPK = self.C*calc_fpk(xp[0:Ncalc], yp[0:Ncalc], self.bx[0:Ncalc], self.by[0:Ncalc], tau0, Nm, Ncalc)
                if Nm > 0:
                    FPK += self.calc_fpk_external(self.bx[0:Nm],self.by[0:Nm],\
                                            sigextxx[0:Nm],sigextyy[0:Nm],sigextxy[0:Nm])
            # fsp2 = np.sum(np.multiply(FPK,np.absolute(np.array([self.bx[0:Nm],\
            #        self.by[0:Nm]]))), axis=0)
            fsp2 = np.sum(np.multiply(FPK,np.array([self.bx[0:Nm],\
                   self.by[0:Nm]])), axis=0)
            hh = fsp*fsp2
            ih = np.nonzero(hh<0.)[0]
            #dislocation with indices ih traversed a minimum and need special treatment
            #reduce speed of dislocation to prevent them from crossing zero force position
            if jc==4:
                self.dx[ih] = 0.
                self.dy[ih] = 0.
                fsp[ih] = 0.
            self.dx[ih] *= 0.5
            self.dy[ih] *= 0.5
            xp[ih] = self.xpos[ih] + self.dx[ih]
            yp[ih] = self.ypos[ih] + self.dy[ih]
            jc += 1
        #update positions according to boundary conditions
        if bc=='fixed':
            self.xpos = np.clip(xp, 0, self.lx)
            self.ypos = np.clip(yp, 0, self.ly)
            bc1 = np.logical_or(self.xpos==0, self.ypos==0)
            bc2 = np.logical_or(self.xpos==self.lx, self.ypos==self.ly)
            ih = np.nonzero(np.logical_or(bc1, bc2))
            fsp[ih] = 0.
            self.dx[ih] = 0.
            self.dy[ih] = 0.
        elif bc=='pbc':
            self.xpos = xp
            self.ypos = yp
            ih = np.nonzero(self.xpos<0.)
            self.xpos[ih] += self.lx
            ih = np.nonzero(self.ypos<0.)
            self.ypos[ih] += self.ly
            ih = np.nonzero(self.xpos>self.lx)
            self.xpos[ih] -= self.lx
            ih = np.nonzero(self.ypos>self.ly)
            self.ypos[ih] -= self.ly
        elif bc=='finite' or bc=='finiteFEM':
            self.xpos = xp
            self.ypos = yp
            # If dislocations exit domain, remove them from simulation:
            bcx = np.logical_or(xp <= 0., xp >= self.lx)
            bcy = np.logical_or(yp <= 0., yp >= self.ly)
            ih = np.nonzero(np.logical_or(bcx, bcy))
            self.xpos = np.delete(self.xpos, ih, axis=0)
            self.ypos = np.delete(self.ypos, ih, axis=0)
            self.sp_inc = np.delete(self.sp_inc, ih, axis=0)
            self.flag = np.delete(self.flag, ih, axis=0)
            self.dx = np.delete(self.dx, ih, axis=0)
            self.dy = np.delete(self.dy, ih, axis=0)
            self.bx = np.delete(self.bx, ih, axis=0)
            self.by = np.delete(self.by, ih, axis=0)
            self.Ntot = self.Ntot - len(ih[0])
            self.Nmob = self.Nmob - len(ih[0])
            self.rho = self.Ntot/(self.lx*self.ly)
            self.rho_m = self.Nmob/(self.lx*self.ly)
            self.dtnuc0 = np.delete(self.dtnuc0, ih, axis=0)
            if sigextxx is not None and sigextyy is not None and sigextxy is not None:
                sigextxx = np.delete(sigextxx, ih, axis=0)
                sigextyy = np.delete(sigextyy, ih, axis=0)
                sigextxy = np.delete(sigextxy, ih, axis=0)
        self.update_source_nucleation(tau0, dt, sxx=sigextxx, syy=sigextyy, sxy=sigextxy)
        #time step control
        if len(nmax)>2:
            dt = np.minimum(np.maximum(self.dt0*0.02, dt*0.9), self.dtmin)    # reduce time step im more than 3 dislocation are fast
        elif dr_max<self.dmax*0.9:
            dt = np.minimum(np.minimum(self.dt0*50, dt*1.1), self.dtmax)     # increase time step if all dislocations are slow
        return fsp, dt

    # relax all dislocation if True, otherwise only mobile dislocations are relaxed
    def relax_disl(self, relax_all=False, ftol=5.e-2, dt=0.02, plot_conf=False,
                   plot_relax=True, arrowson=True):
        # ftot acceptable residual error in force relaxation
        if relax_all:
            Nm = self.Ntot
        else:
            Nm = self.Nmob
        # initialze parameters for relaxation
        fn = 2.*ftol
        nl = 0
        nout = 1000
        fout= int(50000/nout)
        fd = []
        dt = 0.03
        while fn>ftol and nl<50000:
            fsp, dt = self.move_disl(0., 'viscous', dt)  # move dislocations w/o ext. stress, motion is viscous for relaxation
            fn = np.sum(np.absolute(fsp))/Nm
            nl += 1
            if plot_relax and np.mod(nl,fout)==0:
                fd.append(fn)
            if plot_conf and np.mod(nl,5000)==0:
                self.plot_stress(arrowson=arrowson)
                print('Iteration:', nl, ', residual force:',fn)
        self.xpeq = self.xpos  # store equilibrium positions
        self.ypeq = self.ypos
        if plot_conf:
            self.plot_stress(arrowson=arrowson)
            print('Final configuration', nl, fn)
        if plot_relax:
            fd.append(fn)
            fd = np.array(fd)
            plt.semilogy(fd)
            plt.title('Dislocation structure relaxation')
            plt.xlabel('iteration')
            plt.ylabel('PK force norm')
            plt.show()
        return

    # relax all dislocation if True, otherwise only mobile dislocations are relaxed
    def relax_disl_simple(self, relax_all=False, ftol=5.e-2, dt=0.02, plot_conf=False,
                   plot_relax=True, arrowson=True):
        # ftot acceptable residual error in force relaxation
        if relax_all:
            Nm = self.Ntot
        else:
            Nm = self.Nmob
        # initialze parameters for relaxation
        fn = 2.*ftol
        nl = 0
        nout = 1000
        fout= int(50000/nout)
        fd = []
        dt = 0.03
        while fn>ftol and nl<50000:
            fsp, dt = self.move_disl(0., 'viscous', dt)  # move dislocations w/o ext. stress, motion is viscous for relaxation
            fn = np.sum(np.absolute(fsp))/Nm
            nl += 1
            if plot_relax and np.mod(nl,fout)==0:
                fd.append(fn)
            if plot_conf and np.mod(nl,5000)==0:
                self.plot_stress_simple(arrowson=arrowson)
                print('Iteration:', nl, ', residual force:',fn)
        self.xpeq = self.xpos  # store equilibrium positions
        self.ypeq = self.ypos
        if plot_conf:
            self.plot_stress_simple(arrowson=arrowson)
            print('Final configuration', nl, fn)
        if plot_relax:
            fd.append(fn)
            fd = np.array(fd)
            plt.semilogy(fd)
            plt.title('Dislocation structure relaxation')
            plt.xlabel('iteration')
            plt.ylabel('PK force norm')
            plt.show()
        return

    #calculate and plot stress field on grid
    def plot_stress(self, arrowson=True):
        ngp = 150  # number of grid points
        #dx = self.lx/ngp
        #dy = self.ly/ngp
        xp = np.linspace(0, self.lx, ngp)
        yp = np.linspace(0, self.ly, ngp)
        XP, YP = np.meshgrid(xp, yp)
        s11 = np.zeros((ngp,ngp))
        s22 = np.zeros((ngp,ngp))
        s12 = np.zeros((ngp,ngp))
        for i in range(self.Ntot):
            s11 += self.bx[i]*self.sig_xx(XP-self.xpos[i], YP-self.ypos[i])
            s11 += self.by[i]*self.sig_yy(YP-self.ypos[i], XP-self.xpos[i])
            s22 += self.bx[i]*self.sig_yy(XP-self.xpos[i], YP-self.ypos[i])
            s22 += self.by[i]*self.sig_xx(YP-self.ypos[i], XP-self.xpos[i])
            s12 += self.bx[i]*self.sig_xy(XP-self.xpos[i], YP-self.ypos[i])
            s12 += self.by[i]*self.sig_xy(YP-self.ypos[i], XP-self.xpos[i])

        extent = (0, self.lx, 0, self.ly)
        fig, axs  = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        fig.subplots_adjust(hspace=0.2)

        [axs[i].set_xlabel(r'x ($\mu$m)') for i in range(3)]
        [axs[i].set_ylabel(r'y ($\mu$m)') for i in range(3)]
        axs[0].set_title(r'$\sigma_{xx}$ (MPa)')
        axs[1].set_title(r'$\sigma_{yy}$ (MPa)')
        axs[2].set_title(r'$\sigma_{xy}$ (MPa)')
        im = axs[0].imshow(s11, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(s22, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(s12, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        fig.colorbar(im, ax=axs[2])

        # plot markers for dislocations if not too many
        if self.Ntot<10:
            [axs[i].scatter(self.xpos, self.ypos, s=30, c='yellow', marker='o') for i in range(3)]

        #plot arrows for mobile dislocations
        if arrowson == True:
          for i in range(self.Nmob):
              dx = self.dx[i]
              dy = self.dy[i]
              hh = dx*dx + dy*dy
              if hh<self.b0:
                  dx = self.bx[i]
                  dy = self.by[i]
              # axs[0].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
              #          width=0.5, head_length=2, color='#20ff00')
              # axs[1].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
              #          width=0.5, head_length=2, color='#20ff00')
              # axs[2].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
              #          width=0.5, head_length=2, color='#20ff00')
              axs[0].arrow(self.xpos[i], self.ypos[i], dx, dy, \
                          head_width=0.6, width=0.2, color='#20ff00')
              axs[1].arrow(self.xpos[i], self.ypos[i], dx, dy, \
                          head_width=0.6, width=0.2, color='#20ff00')
              axs[2].arrow(self.xpos[i], self.ypos[i], dx, dy, \
                          head_width=0.6, width=0.2, color='#20ff00')
        fig.tight_layout()
        plt.show()

    def plot_stress_mob(self):
        ngp = 150  # number of grid points
        #dx = self.lx/ngp
        #dy = self.ly/ngp
        xp = np.linspace(0, self.lx, ngp)
        yp = np.linspace(0, self.ly, ngp)
        XP, YP = np.meshgrid(xp, yp)
        s11 = np.zeros((ngp,ngp))
        s22 = np.zeros((ngp,ngp))
        s12 = np.zeros((ngp,ngp))
        for i in range(self.Ntot):
            if self.flag[i] == 1:
                continue
            s11 += self.bx[i]*self.sig_xx(XP-self.xpos[i], YP-self.ypos[i])
            s11 += self.by[i]*self.sig_yy(YP-self.ypos[i], XP-self.xpos[i])
            s22 += self.bx[i]*self.sig_yy(XP-self.xpos[i], YP-self.ypos[i])
            s22 += self.by[i]*self.sig_xx(YP-self.ypos[i], XP-self.xpos[i])
            s12 += self.bx[i]*self.sig_xy(XP-self.xpos[i], YP-self.ypos[i])
            s12 += self.by[i]*self.sig_xy(YP-self.ypos[i], XP-self.xpos[i])

        extent = (0, self.lx, 0, self.ly)
        fig, axs  = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        fig.subplots_adjust(hspace=0.2)

        [axs[i].set_xlabel(r'x ($\mu$m)') for i in range(3)]
        [axs[i].set_ylabel(r'y ($\mu$m)') for i in range(3)]
        axs[0].set_title(r'$\sigma_{xx}$ (MPa)')
        axs[1].set_title(r'$\sigma_{yy}$ (MPa)')
        axs[2].set_title(r'$\sigma_{xy}$ (MPa)')
        im = axs[0].imshow(s11, origin='lower', extent=extent, cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(s22, origin='lower', extent=extent, cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(s12, origin='lower', extent=extent, cmap=cm.RdBu)
        fig.colorbar(im, ax=axs[2])

        # plot markers for dislocations if not too many
        if self.Ntot<10:
            [axs[i].scatter(self.xpos, self.ypos, s=30, c='yellow', marker='o') for i in range(3)]

        #plot arrows for mobile dislocations
        # for i in range(self.Nmob):
        #     dx = self.dx[i]
        #     dy = self.dy[i]
        #     hh = dx*dx + dy*dy
        #     if hh<self.b0:
        #         dx = self.bx[i]
        #         dy = self.by[i]
        #     axs[0].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
        #              width=0.5, head_length=2, color='#20ff00')
        #     axs[1].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
        #              width=0.5, head_length=2, color='#20ff00')
        #     axs[2].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
        #              width=0.5, head_length=2, color='#20ff00')
        fig.tight_layout()
        plt.show()

    def plot_stress_simple(self, arrowson=True):
        ngp = 150  # number of grid points
        #dx = self.lx/ngp
        #dy = self.ly/ngp
        xp = np.linspace(0, self.lx, ngp)
        yp = np.linspace(0, self.ly, ngp)
        XP, YP = np.meshgrid(xp, yp)
        s11 = np.zeros((ngp,ngp))
        for i in range(self.Ntot):
            if self.flag[i] == 1:
                continue
            s11 += self.bx[i]*self.sig_xx(XP-self.xpos[i], YP-self.ypos[i])
            s11 += self.by[i]*self.sig_yy(YP-self.ypos[i], XP-self.xpos[i])

        extent = (0, self.lx, 0, self.ly)
        fig, axs  = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.2)

        axs.set_xlabel(r'x ($\mu$m)')
        axs.set_ylabel(r'y ($\mu$m)')
        axs.set_title(r'$\sigma$ (MPa)')

        im = axs.imshow(s11, origin='lower', extent=extent, cmap=cm.RdBu)
        fig.colorbar(im, ax=axs)

        # plot markers for dislocations if not too many
        # if self.Ntot<10:
        #     [axs[i].scatter(self.xpos, self.ypos, s=30, c='yellow', marker='o') for i in range(3)]
        if arrowson == True:
          #plot arrows for mobile dislocations
          for i in range(self.Nmob):
              dx = self.dx[i]
              dy = self.dy[i]
              hh = dx*dx + dy*dy
              if hh<self.b0:
                  dx = self.bx[i]
                  dy = self.by[i]
              # axs.arrow(self.xpos[i], self.ypos[i], dx, dy, head_width=0.1, width=0.1, \
              #           color='#20ff00')
              # axs.arrow(self.xpos[i], self.ypos[i], -dx, -dy, head_width=0.1, width=0.1, \
              #           color='#20ff00')
              axs.arrow(self.xpos[i], self.ypos[i], dx/2, dy/2, head_width=0.01, width=0.1, \
                        color='black')
              axs.arrow(self.xpos[i], self.ypos[i], -dx/2, -dy/2, head_width=0.01, width=0.1, \
                        color='black')
              axs.arrow(self.xpos[i], self.ypos[i], -dy/2, dx/2, head_width=0.01, width=0.1, \
                        color='black')
        fig.tight_layout()
        plt.show()

    #create line plot with Peach Koehler force
    def calc_PKforce(self, hy, ngp=150, x1=0.01, x2=None):
        '''Calculate Peach-Koehler (PK) force along a given plane within the
        current dislocation configuration

        Parameters:
        hy  : float
            y-offset for line of which PK-force is calculate
        npg (optional) : int
            number of grid points
        x1 (optional)  : int
            start point of line plot
        x2 (optional)  : int
            end point of line plot

        #Returns:
        fpk : (npg,)-array
            PK force (units: mN/m)
        xp  : (npg,)-array
            x-positions at which PK force is evaluted (units: micron)
        '''

        if x2 is None:
            x2=self.LX
        Nd = len(self.xpos)  # number of dislocation in group
        xp = np.linspace(x1, x2, num=ngp)
        yp = np.ones(ngp)*hy
        fpk = np.zeros(ngp)
        for i in range(Nd):
            fpk += self.b0*self.bx[i]*self.sig_xy(xp-self.xpos[i], yp-self.ypos[i])
            fpk -= self.b0*self.by[i]*self.sig_xy(yp-self.ypos[i], xp-self.xpos[i])
        return fpk*1000, xp
