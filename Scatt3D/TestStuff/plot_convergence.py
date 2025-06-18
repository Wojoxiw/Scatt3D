import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as c0
import miepython, miepython.field

epsr = 2 - 0j
a = 1e-2
f0 = 10e9
lambda0 = c0/f0
hfactors = np.array([5, 10, 20, 30, 40], dtype=float)
hfactors = np.array([5, 10], dtype=float) # For testing

def ComputeErrors(sol, ref):
    abs_error = np.sqrt(np.sum(np.abs(sol - ref)**2)/len(sol))
    rel_error = np.sqrt(np.sum(np.abs(sol - ref)**2/np.abs(ref)**2)/len(sol))
    return abs_error, rel_error

abs_errors_Eplane = []
rel_errors_Eplane = []
abs_errors_Hplane = []
rel_errors_Hplane = []

for hfactor in hfactors:
    # Compare far field data
    filename = f'ffdata_{hfactor}.dat'
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)
    cut = np.real(data[:,0])
    ffsq_Eplane = np.abs(data[:,1])**2 + np.abs(data[:,2])**2
    ffsq_Hplane = np.abs(data[:,3])**2 + np.abs(data[:,4])**2
    
    # Mie solution far field
    x = 2*np.pi*f0*a/c0
    m = np.sqrt(epsr, dtype=complex)
    mie_E = miepython.i_par(m, x, np.cos(cut*np.pi/180), norm='qsca')*np.pi*a**2
    mie_H = miepython.i_per(m, x, np.cos(cut*np.pi/180), norm='qsca')*np.pi*a**2

    # Compute errors
    abs_error_Eplane, rel_error_Eplane = ComputeErrors(ffsq_Eplane, mie_E)
    abs_error_Hplane, rel_error_Hplane = ComputeErrors(ffsq_Hplane, mie_H)
    abs_errors_Eplane.append(abs_error_Eplane)
    rel_errors_Eplane.append(rel_error_Eplane)
    abs_errors_Hplane.append(abs_error_Hplane)
    rel_errors_Hplane.append(rel_error_Hplane)
        
    # Plotting comparison
    plt.figure()
    plt.semilogy(cut, ffsq_Eplane, label='E plane')
    plt.semilogy(cut, ffsq_Hplane, label='H plane')
    plt.semilogy(cut, mie_E, ':', label='Mie E')
    plt.semilogy(cut, mie_H, ':', label='Mie H')
    plt.xlabel('Angle (degrees)')
    plt.grid()
    plt.legend(loc='best')
    plt.title(f'lambda/h = {hfactor}')

    # Compare near field data
    filename = f'nfdata_{hfactor}.dat'
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)
    line = np.real(data[:,1]) # Line in the y direction
    Ex = data[:,3]
    Ey = data[:,4]
    Ez = data[:,5]

    # Mie solution near field
    m = np.sqrt(epsr, dtype=complex)
    abcd = miepython.core.coefficients(m, 2*np.pi/lambda0*a, internal=True)
    points = data[:,0:3].real
    Ex_inc = np.exp(-2j*np.pi/lambda0*points[:,2])
    Ex_inc[np.abs(points[:,1])>a] = 0
    Ex_mie = []
    Ey_mie = []
    Ez_mie = []
    for p in points:
        r = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        theta = np.arccos(p[2]/r)
        phi = np.atan2(p[1], p[0])
        E_r, E_theta, E_phi = miepython.field.e_near(abcd, lambda0, 2*a, m, r, theta, phi)
        Ex_ = E_r*np.sin(theta)*np.cos(phi) + E_theta*np.cos(theta)*np.cos(phi) - E_phi*np.sin(phi)
        Ey_ = E_r*np.sin(theta)*np.sin(phi) + E_theta*np.cos(theta)*np.sin(phi) + E_phi*np.cos(phi)
        Ez_ = E_r*np.cos(theta) - E_theta*np.sin(theta)
        Ex_mie.append(Ex_)
        Ey_mie.append(Ey_)
        Ez_mie.append(Ez_)
    Ex_mie = np.array(Ex_mie)
    Ey_mie = np.array(Ey_mie)
    Ez_mie = np.array(Ez_mie)

    # Read Feko solution
    #data = np.genfromtxt('near_field_feko.dat', dtype=None, delimiter='\t', skip_header=2)
    #print(data)
    #y_feko = data[:,0]
    #Ex_feko = data[:,1] + 1j*data[:,2]
    #Ex_inc_feko = 1
    
    plt.figure()
    plt.plot(line, Ex.real, label='Re(Ex)')
    plt.plot(line, Ex.imag, label='Im(Ex)')
#    plt.plot(line, -Ex_mie.real-Ex_inc.real, '--', label='Re(Ex_mie)')
#    plt.plot(line, Ex_mie.imag+Ex_inc.imag, '--', label='Im(Ex_mie)')
    #plt.plot(y_feko, Ex_feko.real - Ex_inc_feko.real, '--', label='Re(Ex_feko)')
    #plt.plot(y_feko, Ex_feko.imag - Ex_inc_feko.imag, '--', label='Im(Ex_feko)')
#    plt.plot(line, Ex_inc.real, ':', label='Re(Ex_inc)')
#    plt.plot(line, Ex_inc.imag, ':', label='Im(Ex_inc)')
    plt.xlabel('Coordinate')
    plt.grid()
    plt.legend(loc='best')
    plt.title(f'lambda/h = {hfactor}')
    
plt.figure()
plt.loglog(hfactors, abs_errors_Eplane, label='abs')
plt.loglog(hfactors, rel_errors_Eplane, label='rel')
plt.xlabel('lambda/h')
plt.legend(loc='best')
plt.grid()
plt.title('E plane')

plt.figure()
plt.loglog(hfactors, abs_errors_Hplane, label='abs')
plt.loglog(hfactors, rel_errors_Hplane, label='rel')
plt.xlabel('lambda/h')
plt.legend(loc='best')
plt.grid()
plt.title('H plane')

plt.show()
