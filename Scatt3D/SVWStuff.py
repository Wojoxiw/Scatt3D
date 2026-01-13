## Spherical vector wave stuff for use in Scatt3D - based on the stuff in SNFAM_Stuff
from scipy.constants import c, epsilon_0, mu_0, k, elementary_charge, m_p
import numpy as np
import scipy
from math import pi, factorial
from scipy.special import spherical_jn, spherical_yn, lpmv, binom
from scipy.integrate import dblquad
from wigners import wigner_3j
import matplotlib.pyplot as plt
from numpy.random._examples.cffi.extending import vals
import os

eta_0 = np.sqrt(mu_0/epsilon_0) ##impedance of free space
eta_0_h = np.sqrt(epsilon_0/mu_0) ## specific admittance, definition from Hansen [3] and EuCAP course slides
eps = 1e-6 ## adjust angles with this to avoid singularities with trig functions.

## make sure angles are in radians here


#####################################################
# Mathematical functions to use
#####################################################
## following Hansen's single index convention [3], but equations from 'Theory and Practice' [1]. B1 has s=1, B2 has s=2
##

def jFromsmn(s,m,n): # from [3] appendix A1
    return (s+2*( n*(n+1) + m-1 )) - 1 ## -1 to go from j to python index

def smnFromj(j): # from [3] appendix A1
    j = j+1 ##to translate from Python starting at index 0
    if(j%2 == 0):
        s = 2
    else:
        s = 1
    n = int(np.floor(np.sqrt((j - s)/2 + 1)))
    m = int((j-s)/2 + 1 -n*(n+1))
    return s, m, n

def JfromN(N):
    return 2*N*(N+2)

## going between transmission and receiving, see (3.22), or (2.107) in Hansen [3], also (A2.16, 3.21)... Daniel's notes since probe used as AUT is rotated pi around y-axis to point to probe
def RfromT(T): ## assuming here that the antenna is reciprocal:
    R = np.zeros(len(T), dtype=complex)
    
    for j in range(len(T)): ## how Daniel did it
        s, m, n = smnFromj(j)
        R[j] = (-1)**(n) * T[j]
    
    #===========================================================================
    # for j in range(len(T)): ## (2.107) - will not work/give bad results
    #     s, m, n = smnFromj(j)
    #     R[j] = (-1)**(m) * T[jFromsmn(s, -m, n)]
    #===========================================================================
        
    return R

def delta(a,b): ## kronecker delta
    if(a==b):
        return 1
    else:
        return 0

def SMCsHertzianDipole(J, rotatedForMeas = False): ##based on section 2.3.4 in [2], calculates the SMCs with N modes or J coefficients. They are all zero except when s=2,m=0,n=1 (j = 4)
                                    ## since this is for a z-directed dipole, should rotate these coefficients for i.e. the actual probe, which is in the x-y plane

    T = np.zeros((J), dtype = complex) ## the SMCs
    if(rotatedForMeas): ## Hertzian dipole R^p, rotated to take place of the probe. Seems to then have strange polarizations...
        T[jFromsmn(2, 1, 1)] = -1/np.sqrt(2)
        T[jFromsmn(2, -1, 1)] = 1/np.sqrt(2)
    else:
        for j in range(np.size(T)):
            if(j == 4): 
                T[j-1] = 1 ## j-1 to go from j to python index
    return T

def calcTsProbeCorr(data, freq): ## calculates Tsmn as in section (4.3.2) with self.ProbeRs probe receiving coefficients (Rs), takes the data and the frequency to calc for
    print('Calculating Tsmn with probe correction...')
    Tsmn = np.zeros(data.J, dtype=complex)
    kA = data.A*2*pi*data.fs[freq]/c
    Cs = getCs(data.J, kA) ## translation coefficients
    
    Rs = data.ProbeRs
    if(np.size(Rs) < data.J): ## since the probe is smaller than the AUT - set extra coefficients to zero
        Rs = np.zeros(data.J, dtype=complex)
        Rs[0:np.size(data.ProbeRs)] = data.ProbeRs
    
    Ps = np.dot(Cs, Rs)/2 ## probe response constants, (4.39)
    Ss = data.S21_sphere[freq] ## probe pol (theta, then phi), theta, and phi angle
    thetavec = data.thetavec_sphere*pi/180
    phivec = data.phivec_sphere*pi/180
    dPhi = data.phiSpacing*pi/180
    dTheta = data.thetaSpacing*pi/180
    
    ### remove any near-zero values to avoid dividing by zero
    thetavec[np.abs(thetavec) < eps] = eps
    thetavec[np.abs(thetavec - pi) < eps] = pi - eps
    
    ## the phi integral
    wm_theta = np.fft.fft(Ss[0], axis=1)*dPhi ## chi = 0, theta-hat part
    wm_phi = np.fft.fft(Ss[1], axis=1)*dPhi ## chi = pi/2, phi-hat part
    ### 4.61 and on, the chi integral (4.65, 4.66)
    wmum_p1 = (wm_theta - 1j*wm_phi)/2 ## w_mum, +1 part
    wmum_n1 = (wm_theta + 1j*wm_phi)/2 ## w_mum, -1 part
    
    ## and solve (4.133, 4.134, or 4.53, 4.54) (spacing of 2 since s=1 and s=2 solved together)
    for j in range(0, data.J, 2):
        s, m, n = smnFromj(j)
        P = np.array([ [Ps[jFromsmn(1, 1, n)], Ps[jFromsmn(2, 1, n)]], [Ps[jFromsmn(1, -1, n)], Ps[jFromsmn(2, -1, n)]] ]) ## each value of P
        ### theta integral, (4.55)
        wnmum_p1 = (2*n + 1)/2*np.sum(wmum_p1[:,m]*dnmum(n, 1, m, thetavec)*np.sin(thetavec)*dTheta)
        wnmum_n1 = (2*n + 1)/2*np.sum(wmum_n1[:,m]*dnmum(n, -1, m, thetavec)*np.sin(thetavec)*dTheta)
        
        W = np.array([wnmum_p1, wnmum_n1])
        T = np.linalg.solve(P, W) ## this gets T_1mn and T_2mn
        Tsmn[jFromsmn(1, m, n)] = T[0]
        Tsmn[jFromsmn(2, m, n)] = T[1]
        
    return Tsmn

def calcTsNoProbeCorr(data, freq): ## calculates Tsmn as in (4.30) assuming no probe correction (electric dipole probe), takes the data and the frequency
                                ## data as in SNFAM Stuff
    J = data.J
    A = data.A
    f = data.fs[freq]
    k = 2*pi*f / c

    ## method, using NFTs and np summing to be much faster:
    Q = np.zeros(J, dtype=complex)
    Es = data.S21_sphere[freq]
    thetavec = data.thetavec_sphere*pi/180
    phivec = data.phivec_sphere*pi/180
    Nphi = len(phivec)
    dphi = data.phiSpacing*pi/180
    dtheta = data.thetaSpacing*pi/180
           
    EDFT = np.fft.fft(Es, axis = 2)*dphi ## compute phi integral as a DFT
    #EDFT = Nphi*np.fft.ifft(Es, axis = 2)*dphi ## compute phi integral as a DFT
        
    ### remove any near-zero values to avoid dividing by zero
    if np.isscalar(thetavec):
        if np.abs(thetavec) < eps:
            thetavec = eps
        elif np.abs(thetavec - np.pi) < eps:
            thetavec = np.pi - eps
    else:
        thetavec[np.abs(thetavec) < eps] = eps
        thetavec[np.abs(thetavec - pi) < eps] = pi - eps
        
    for j in range(J):
        if (j%int(J/12.67))== 0:
            print(f'Computing SMCs, j = {j+1} / {J}')
        s, m, n = smnFromj(j)
        F_r, F_theta, F_phi = Fcsmn(3, s, -m, n, A, k, thetavec, 0)
        prefactor = 2/(np.sqrt(6*pi))* (-1)**m / (Rc_sn(k*A, 3, s, n)**2)
        Q[j] = prefactor * np.sum( F_theta*EDFT[0][:,m]*np.sin(thetavec)*dtheta + F_phi*EDFT[1][:,m]*np.sin(thetavec)*dtheta ) ### the theta integration
    return Q

def spherical_h1(n, z, derivative=False):
    return(spherical_jn(n, z, derivative) + 1j*spherical_yn(n, z, derivative))

def spherical_h2(n, z, derivative=False):
    return(spherical_jn(n, z, derivative) - 1j*spherical_yn(n, z, derivative))

# Functions of z = kr
def zc_n(kr, c, n, derivative=False): ## from (2.10)
    if c == 1:
        z_func = spherical_jn(n, kr, derivative)
    elif c == 2:
        z_func = spherical_yn(n, kr, derivative)
    elif c == 3:
        z_func = spherical_h1(n, kr, derivative)
    else:
        z_func = spherical_h2(n, kr, derivative)
    return(z_func)

def Rtildecgamma_sn(kr, c, gamma, s, n): ## from (4.9) (for 4.17)
    if(s==1):
        return Rc_sn(kr, c, s, n)*Rc_sn(kr, gamma, s, n)
    else:
        return Rc_sn(kr, c, s, n)*Rc_sn(kr, gamma, s, n) + n*(n+1)* (zc_n(kr, c, n)/kr) * (zc_n(kr, c, gamma)/kr)

def Rc_sn(kr, c, s, n): ## from (A1.6)
    if(s==1):
        return zc_n(kr, c, n)
    else:
        return oneoverkrdkrzdkr(kr, c, n)

def normlpmv(m, n, x): ## normalized associated legendre function, as in (A1.25)
    if(m+n > 100): ## to avoid an overflow error
        return 0
    else:
        factor = np.sqrt( (2*n+1)/2 * factorial(n-m)/factorial(n+m) )
        return lpmv(m,n,x) * factor *(-1)**m ## just includes the normalization factor, and ## lpmv is the associated legendre function, defined in Hansen without the -1**m factor, which is included in the scipy implementation
    
def oneoverkrdkrzdkr(kr, c, n):
    return zc_n(kr, c, n, derivative = True) + zc_n(kr, c, n)/kr ## using derivatives instead
    #return (n+1) * zc_n(kr, n, n)/(kr) - zc_n(kr, c, n+1) ## from (A1.9)
    #return zc_n(kr, c, n-1) + n*zc_n(kr, c, n)/kr ## from (A1.8)
    
def Pbarm_n(m, n, costheta): ## normalized associated legendre function as in (A1.25)
    return normlpmv(m, n, costheta) ## lpmv is the associated legendre function, defined in Hansen without the -1**m factor, which is included in the scipy implementation

def dPbar(m, n, theta): ## dPbar(costheta)/dtheta, as in (A1.34b). ## including factor to go from P to Pbar
    costheta = np.cos(theta)
    if(m+n > 100): ## to avoid an overflow error
        normfactor = 0
    else:
        normfactor = np.sqrt( (2*n+1)/2 * factorial(n-m) / factorial(n+m) ) ## to transport P to Pbar
    if(m==0):
        return 1*lpmv(1, n, costheta) * normfactor
    else:
        sintheta = np.sin(theta)
        return  -( (n-m+1)*(n+m)*lpmv(m-1,n,costheta) + m*costheta/sintheta*lpmv(m,n,costheta) ) * normfactor *(-1)**m ##P(m+1) term changed out using recurrance relation A1.32, since m+1 can be greater than n
    
def FieldValue(Q, k, r, theta, phi, c=1): ## Finds the E-field at some point(s) [r, theta, phi], from given Qs, from (A1.1)
    Er, Etheta, Ephi = 0j, 0j, 0j
    for j in range(len(Q)):
        s, m, n = smnFromj(j)
        F = Fcsmn(c, s, m, n, r, k, theta, phi)
        Er = Er + Q[j]*F[0]
        Etheta = Etheta + Q[j]*F[1]
        Ephi = Ephi + Q[j]*F[2]
    return Er, Etheta, Ephi
    
def Fcsmn(c, s, m, n, A, k, theta, phi): ## spherical wave function, from (A1.45)
    if(m==0):
        mpart = 1
    else:
        mpart = (-1*m/np.abs(m))**m
    prefactor = 1/( np.sqrt(2*pi*n*(n+1)) ) * mpart * np.exp(1j*m*phi)
    costheta = np.cos(theta)
    if(s==1):
        thetaPart = prefactor* zc_n(k*A, c, n) * 1j*m*Pbarm_n(np.abs(m), n, costheta) / (np.sin(theta))  ## theta-hat part
        phiPart = prefactor* -1* zc_n(k*A, c, n) * dPbar(np.abs(m), n, theta) ##phi-hat part
        return [thetaPart*0, thetaPart, phiPart]
    else:
        rPart = prefactor* n*(n+1)/(k*A) * zc_n(k*A, c, n) * Pbarm_n(np.abs(m), n, costheta) ## r-hat part
        thetaPart = prefactor* oneoverkrdkrzdkr(k*A, c, n) * dPbar(np.abs(m), n, theta) ## theta-hat part
        phiPart = prefactor* oneoverkrdkrzdkr(k*A, c, n) * 1j*m*Pbarm_n(np.abs(m), n, costheta) / (np.sin(theta)) ##phi-hat part
        return [rPart, thetaPart, phiPart]


def dnmum(n, mu, m, theta): ## rotation coefficients, from Hansen (A2.5) ##for the jacobi, possibly the top two arguments have to be greater than -1, so use if statements to convert them if not
    ## check arguments for Jacobi polynomial - if bad, switch using symmetries (A2.8, A2.9)
    if(mu - m > -1 and mu + m > -1): ## args are good
        d = find_dnmum(n, mu, m, theta)
    elif(mu - m <= -1 and mu + m > -1): ## swap m and mu
        d = find_dnmum(n, m, mu, theta)*(-1)**(mu + m)
    elif(mu - m <= -1 and mu + m <= -1): ## make m and mu negative
        d = find_dnmum(n, -mu, -m, theta)*(-1)**(mu + m)
    elif(mu - m > -1 and mu + m <= -1): ## swap m and mu + make them negative
        d = find_dnmum(n, -m, -mu, -theta)*(-1)**(mu + m) ## Daniel has this factor, but it seems like it should be cancelled out - maybe has no effect?
    
    return d
def find_dnmum(n, mu, m, theta): ## to allow changing mu, m, etc
    prefactor = np.sqrt( (factorial(n+mu)*factorial(n-mu)) / (factorial(n+m)*factorial(n-m)) ) * np.cos(theta/2)**(mu+m)*np.sin(theta/2)**(mu-m)
    return prefactor * scipy.special.eval_jacobi(n-mu, mu-m, mu+m, np.cos(theta))

def Csn3sigmunu(s,n,sigma,nu,mu,kA): ## from Hansen (A3.3), assuming positive kA
    prefactor = np.sqrt( (2*n+1)*(2*nu+1) / (n*(n+1)*nu*(nu+1)) ) * np.sqrt(factorial(nu+mu)*factorial(n-mu) / (factorial(nu-mu)*factorial(n+mu)) ) * (-1)**(mu)*1/2*(1j)**(n-nu)
    sum = 0
    for p in range(np.abs(n-nu), n+nu+1):
        sum += (1j)**(-p) * (delta(s,sigma)*( n*(n+1) + nu*(nu+1) - p*(p+1) ) + delta(3-s,sigma)*2j*mu*kA ) * a(mu,n,nu,p)*zc_n(kA, 3, p)
    return prefactor*sum

def a(mu,n,nu,p): ## 'linearization coefficients', from (A3.6)
    return (2*p+1)*np.sqrt( factorial(n+mu)*factorial(nu-mu) / (factorial(n-mu)*factorial(nu+mu)))*wigner_3j(n, nu, p, 0, 0, 0)*wigner_3j(n, nu, p, mu, -mu, 0)

def getCs(J, kA, calcNew = False): ## computes the translation coefficients above, or loads them if already calculated for a given J, kA
    fF = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/savedCs/'
    file = fF+f'J{J}kA{kA:.5f}.npz'
    if(os.path.isfile(file) and not calcNew):
        print('Importing previous translation coefficents...')
        C = np.load(file)['Cs']
    else:
        C = np.zeros((J, J), dtype=complex)
        for j in range(J):
            if np.mod(j, int(J/9.2)) == 0:
                print(f'Computing Csn3sigmunu, j = {j} / {J}')
            s, m, n = smnFromj(j)
            for j2 in range(J):
                sigma, mu, nu = smnFromj(j2)
                if mu == m:
                    C[j,j2] = Csn3sigmunu(s,n,sigma,nu,mu,kA)
        np.savez(file,Cs = C)
    return C

def Ksmn(s,m,n,theta,phi): ## Ksmn, from Hansen (A1.59, A1.60)
    if(m==0):
        mPart = 1
    else:
        mPart = (-1*np.abs(m)/m)**m
    if(s==1):
        prefactor = np.sqrt(2/(n*(n+1))) * mPart * (-1j)**(n+1) * np.exp(-1j*m*phi) ## either +j or -j, causes a rotation
        Kthet = 1j*m/np.sin(theta) * Pbarm_n(np.abs(m), n, np.cos(theta)) ## theta part K
        Kphi = -1 * dPbar(np.abs(m), n, theta) ## phi part K
    else:
        prefactor = np.sqrt(2/(n*(n+1))) * mPart * (-1j)**(n) * np.exp(-1j*m*phi) ## either +j or -j, causes a rotation
        Kthet = dPbar(np.abs(m), n, theta) ## theta part K
        Kphi = 1j*m/np.sin(theta) * Pbarm_n(np.abs(m), n, np.cos(theta)) ## phi part K

    return Kthet*prefactor, Kphi*prefactor

def findFarField(Ts, thetas, phis):
    '''
    Finds far-field at some theta, phi using transmission coefficients, as in (2.182, 2.180), Ks as in (A1.59, A1.60)
    Returns array of theta- and phi- pol Es
    If theta and phi vectors are same length, calculate 1-D answer rather than use outer products (returns array of [2, len(angles)]). Otherwise, [2, len(thetas), len(phis)]    
    :param Ts: The coefficients
    :param thetas: Vector of theta angles
    :param phis: Vector of phi angles
    '''
    J = len(Ts)
    ### remove any near-zero values to avoid dividing by zero
    if np.isscalar(thetas):
        if np.abs(thetas) < eps:
            thetas = eps
        elif np.abs(thetas - np.pi) < eps:
            thetas = np.pi - eps
    else:
        thetas[np.abs(thetas) < eps] = eps
        thetas[np.abs(thetas - pi) < eps] = pi - eps
       
    if(len(thetas)==len(phis)): ## to get data for cut plotting, or arbitrary angles
        bigK = np.zeros((2, len(thetas)), dtype=complex) ## pol (theta, then phi), theta, and phi angle
        for j in range(J):
            if np.mod(j, int(J/3.2)) == 0:
                print(f'Computing farfield cut, j = {j} / {J}')
            s, m, n = smnFromj(j)
            bigK += Ts[j]*Ksmn(s, m, n, thetas, phis)
        return bigK/np.sqrt(4*pi*eta_0_h)
    else: ## to get data for sphere plotting
        bigK = np.zeros((2, len(thetas), len(phis)), dtype=complex) ## pol (theta, then phi), theta, and phi angle
        for j in range(J):
            if np.mod(j, int(J/4.2)) == 0:
                print(f'Computing farfield top sphere, j = {j} / {J}')
            s, m, n = smnFromj(j)
               
            bigK += Ts[j]*Ksmn(s, m, n, thetas, phis)
        return bigK/np.sqrt(4*pi*eta_0_h)


### REFERENCES

#[1]: Theory and Practice of Modern Antenna Range Measurements 2nd Expanded Edition, Volume 2 (2020)

#[2] https://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html

#[3]: Spherical near-field antenna measurements, Hansen (1988)

#[4]: Fully Probe-Corrected Near-Field Far-Field Transformations With Unknown Probe Antennas, Paulus (2023)