# encoding: utf-8
# Compute scattering from an isotropic sphere using Hansen's spherical
# vector waves. Make some plots of the fields.
#
# Daniel Sjöberg, 2026-01-09. I slightly modified this

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as c0, epsilon_0 as eps0, mu_0 as mu0
import SVWStuff as svw
eta = np.sqrt(eps0/mu0)

def ComputeQcoefficients(k, a, epsr, mur, theta0, phi0, E0_theta, E0_phi):
    ka = k*a
    N = int(ka + 4.05*ka**(1/3) + 2) # Wiscombe criterion
    J = 2*N*(N + 2)
    Qinc = np.zeros(J, dtype=complex)
    Qsca = np.zeros(J, dtype=complex)
    Qint = np.zeros(J, dtype=complex)

    for j in np.arange(J):
        s, m, n = svw.smnFromj(j)
        Ksmn_theta, Ksmn_phi = svw.Ksmn(s, -m, n, theta0, phi0)
        Qinc[j] = np.sqrt(eta)/k*(-1)**m*np.sqrt(4*np.pi)*1j*(E0_theta*Ksmn_theta + E0_phi*Ksmn_phi)

        Rsn1ka = svw.Rc_sn(ka, 1, s, n)
        Rsn3ka = svw.Rc_sn(ka, 3, s, n)
        Rsn1k1a = svw.Rc_sn(np.sqrt(epsr*mur)*ka, 1, s, n)
        Rqn1ka = svw.Rc_sn(ka, 1, 3-s, n)
        Rqn3ka = svw.Rc_sn(ka, 3, 3-s, n)
        Rqn1k1a = svw.Rc_sn(np.sqrt(epsr*mur)*ka, 1, 3-s, n)
        Qsca[j] = -(np.sqrt(epsr)*Rsn1ka*Rqn1k1a - np.sqrt(mur)*Rqn1ka*Rsn1k1a)/(np.sqrt(epsr)*Rsn3ka*Rqn1k1a - np.sqrt(mur)*Rqn3ka*Rsn1k1a)*Qinc[j]
        Qint[j] = (epsr*mur)**(-0.25)*(Rsn1ka*Rqn3ka - Rqn1ka*Rsn3ka)/(np.sqrt(mur)*Rsn1k1a*Rqn3ka - np.sqrt(epsr)*Rqn1k1a*Rsn3ka)*Qinc[j]

    return Qinc, Qsca, Qint

def ComputeField(Qinc, Qsca, Qint, k, a, pos, epsr, mur):
    """Compute the electric field in points given by a vector of Cartesian positions, pos.shape = (3, Np)."""
    J = len(Qinc)
    Np = pos.shape[1]
    Er = np.zeros(Np, dtype=complex)
    Etheta = np.zeros(Np, dtype=complex)
    Ephi = np.zeros(Np, dtype=complex)    
    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2) + 1e-10 ## to avoid dividing by zero
    theta = np.arccos(pos[2]/r)
    phi = np.arctan2(pos[1], pos[0])
    idx1 = r<=a  # Indices of positions inside (or on) the sphere
    idx2 = ~idx1 # Indices of positions outside the sphere
    
    phi[theta>np.pi] = phi[theta>np.pi]+np.pi
    
    Er[idx1], Etheta[idx1], Ephi[idx1] = epsr**0.25*mur**0.75*k/np.sqrt(eta)*np.array(svw.FieldValue(Qint, k, np.sqrt(epsr*mur)*r[idx1], theta[idx1], phi[idx1], c=1))
    Er[idx2], Etheta[idx2], Ephi[idx2] = k/np.sqrt(eta)*(np.array(svw.FieldValue(Qinc, k, r[idx2], theta[idx2], phi[idx2], c=1)) + np.array(svw.FieldValue(Qsca, k, r[idx2], theta[idx2], phi[idx2], c=3)))
    
    Ex = Er*np.sin(theta)*np.cos(phi) + Etheta*np.cos(theta)*np.cos(phi) - Ephi*np.sin(phi)
    Ey = Er*np.sin(theta)*np.sin(phi) + Etheta*np.cos(theta)*np.sin(phi) + Ephi*np.cos(phi)
    Ez = Er*np.cos(theta) - Etheta*np.sin(theta)
    E = np.array([Ex, Ey, Ez])

    return E

if __name__ == '__main__':
    a = 0.00989
    k = 209
    epsr = 2 - 0.02j
    mur = 1 + 0j
    theta0 = 1e-10
    phi0 = 0
    E0_theta = 1
    E0_phi = 0

    Qinc, Qsca, Qint = ComputeQcoefficients(k, a, epsr, mur, theta0, phi0, E0_theta, E0_phi)

    Np = 100
    vec = np.linspace(-2*a, 2*a, Np)
    
    x = 0*vec
    y = 0*vec
    z = vec
    pos = np.array([x, y, z])
    E = ComputeField(Qinc, Qsca, Qint, k, a, pos, epsr, mur)
    E = np.conjugate(E) # Convert to time convention exp(+j*omega*t)
    plt.figure()
    plt.plot(vec, np.real(E[0]), '-', label='Re(Ex)')
    plt.plot(vec, np.imag(E[0]), '--', label='Im(Ex)')
    plt.xlabel('z (m)')
    plt.legend()
    plt.title(f'a={a}m, ka={k*a:.2}')

    x = vec
    y = 0*vec
    z = 0*vec
    pos = np.array([x, y, z])
    E = ComputeField(Qinc, Qsca, Qint, k, a, pos, epsr, mur)
    E = np.conjugate(E) # Convert to time convention exp(+j*omega*t)
    plt.figure()
    plt.plot(vec, np.real(E[0]), '-', label='Re(Ex)')
    plt.plot(vec, np.imag(E[0]), '--', label='Im(Ex)')
    plt.xlabel('x (m)')
    plt.legend()
    plt.title(f'a={a}m, ka={k*a:.2}')

    plt.show()
    
