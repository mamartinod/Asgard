"""A selection of useful functions for optics, especially Fourier optics. 

Abridged from mikeireland/opticstools.
"""

from __future__ import print_function, division
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    nthreads=6 
except:
    nthreads=0

#On load, create a quick index of the first 100 Zernike polynomials, according to OSA/ANSI:
MAX_ZERNIKE=105
ZERNIKE_N = np.empty(MAX_ZERNIKE, dtype=int)
ZERNIKE_M = np.empty(MAX_ZERNIKE, dtype=int)
ZERNIKE_NORM = np.ones(MAX_ZERNIKE)
n=0
m=0
for z_ix in range(0,MAX_ZERNIKE):
    ZERNIKE_N[z_ix] = n
    ZERNIKE_M[z_ix] = m
    if m==0:
        ZERNIKE_NORM[z_ix] = np.sqrt(n+1)
    else:
        ZERNIKE_NORM[z_ix] = np.sqrt(2*(n+1))
    if m==n:
        n += 1
        m = -n
    else:
        m += 2

def zernike(sz, coeffs=[0.,0.,0.], diam=None, rms_norm=False):
    """A zernike wavefront centered on the *middle*
    of the python array.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    coeffs: float array
        Zernike coefficients, starting with piston.
    diam: float
        Diameter for normalisation in pixels.      
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    if not diam:
        diam=sz
    rr = np.sqrt(xy[0]**2 + xy[1]**2)/(diam/2)
    phi = np.arctan2(xy[0], xy[1])
    n_coeff = len(coeffs)
    phase = np.zeros((sz,sz))
    #Loop over each zernike term.
    for coeff,n,m_signed,norm in zip(coeffs,ZERNIKE_N[:n_coeff], ZERNIKE_M[:n_coeff], ZERNIKE_NORM[:n_coeff]):
        m = np.abs(m_signed)
        #Reset the term.
        term = np.zeros((sz,sz))
        
        #The "+1" is to make an inclusive range.
        for k in range(0,(n-m)//2+1):
            term += (-1)**k * np.math.factorial(n-k) / np.math.factorial(k)/\
                np.math.factorial((n+m)/2-k) / np.math.factorial((n-m)/2-k) *\
                rr**(n-2*k)
        if m_signed < 0:
            term *= np.sin(m*phi)
        if m_signed > 0:
            term *= np.cos(m*phi)
            
        #Add to the phase
        if rms_norm:
            phase += term*coeff*norm
        else:
            phase += term*coeff

    return phase

def zernike_wf(sz, coeffs=[0.,0.,0.], diam=None, rms_norm=False):
    """A zernike wavefront centered on the *middle*
    of the python array. Amplitude of coefficients
    normalised in radians.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    coeffs: float array
        Zernike coefficients, starting with piston.
    diam: float
        Diameter for normalisation in pixels.      
    """
    return np.exp(1j*zernike(sz, coeffs, diam, rms_norm))


def kmf(sz, L_0=np.inf, r_0_pix=None):
    """This function creates a periodic wavefront produced by Kolmogorov turbulence. 
    It SHOULD normalised so that the variance at a distance of 1 pixel is 1 radian^2.
    To scale this to an r_0 of r_0_pix, multiply by sqrt(6.88*r_0_pix**(-5/3))
    
    The value of 1/15.81 in the code is (I think) a numerical approximation for the 
    value in e.g. Conan00 of np.sqrt(0.0229/2/np.pi)
    
    Parameters
    ----------
    sz: int
        Size of the 2D array
        
    l_0: (optional) float
        The von-Karmann outer scale. If not set, the structure function behaves with
        an outer scale of approximately half (CHECK THIS!) pixels. 
   
    r_0_pix: (optional) float
	The Fried r_0 parameter in units of pixels.
 
    Returns
    -------
    wavefront: float array (sz,sz)
        2D array wavefront, in units of radians. i.e. a complex electric field based
        on this wavefront is np.exp(1j*kmf(sz))
    """
    xy = np.meshgrid(np.arange(sz/2 + 1)/float(sz), (((np.arange(sz) + sz/2) % sz)-sz/2)/float(sz))
    dist2 = np.maximum( xy[1]**2 + xy[0]**2, 1e-12)
    ft_wf = np.exp(2j * np.pi * np.random.random((sz,sz//2+1)))*dist2**(-11.0/12.0)*sz/15.81
    ft_wf[0,0]=0
    if r_0_pix is None:
        return np.fft.irfft2(ft_wf)
    else:
        return np.fft.irfft2(ft_wf) * np.sqrt(6.88*r_0_pix**(-5/3.))
        
def join_bessel(U,V,j):
    """In order to solve the Laplace equation in cylindrical co-ordinates, both the
    electric field and its derivative must be continuous at the edge of the fiber...
    i.e. the Bessel J and Bessel K have to be joined together. 
    
    The solution of this equation is the n_eff value that satisfies this continuity
    relationship"""
    W = np.sqrt(V**2 - U**2)
    return U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
    
def neff(V, accurate_roots=True):
    """For a cylindrical fiber, find the effective indices of all modes for a given value 
    of the fiber V number. 
    
    Parameters
    ----------
    V: float
        The fiber V-number.
    accurate_roots: bool (optional)
        Do we find accurate roots using Newton-Rhapson iteration, or do we just use a 
        first-order linear approach to zero-point crossing?"""
    delu = 0.04
    numu = int(V/delu)
    U = np.linspace(delu/2,V - 1e-6,numu)
    W = np.sqrt(V**2 - U**2)
    all_roots=np.array([])
    n_per_j=np.array([],dtype=int)
    n_modes=0
    for j in range(int(V+1)):
        f = U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
        crossings = np.where(f[0:-1]*f[1:] < 0)[0]
        roots = U[crossings] - f[crossings]*( U[crossings+1] - U[crossings] )/( f[crossings+1] - f[crossings] )
        if accurate_roots:
            for i in range(len(crossings)):
                roots[i] = optimize.brenth(join_bessel, U[crossings[i]], U[crossings[i]+1], args=(V,j))
        
#roots[i] = optimize.newton(join_bessel, root, args=(V,j))
#                except:
#                    print("Problem finding root, trying 1 last time...")
#                    roots[i] = optimize.newton(join_bessel, root + delu/2, args=(V,j))
        #import pdb; pdb.set_trace()
        if (j == 0): 
            n_modes = n_modes + len(roots)
            n_per_j = np.append(n_per_j, len(roots))
        else:
            n_modes = n_modes + 2*len(roots)
            n_per_j = np.append(n_per_j, len(roots)) #could be 2*length(roots) to account for sin and cos.
        all_roots = np.append(all_roots,roots)
    return all_roots, n_per_j
 
def mode_2d(V, r, j=0, n=0, sampling=0.3,  sz=1024):
    """Create a 2D mode profile. 
    
    Parameters
    ----------
    V: Fiber V number
    
    r: core radius in microns
    
    sampling: microns per pixel
    
    n: radial order of the mode (0 is fundumental)
    
    j: azimuthal order of the mode (0 is pure radial modes)
    TODO: Nonradial modes."""
    #First, find the neff values...
    u_all,n_per_j = neff(V)
    
    #Unsigned 
    unsigned_j = np.abs(j)
    th_offset = (j<0) * np.pi/2
    
    #Error check the input.
    if n >= n_per_j[unsigned_j]:
        print("ERROR: this mode is not bound!")
        raise UserWarning
    
    # Convert from float to be able to index
    sz = int(sz)
    
    ix = np.sum(n_per_j[0:unsigned_j]) + n
    U0 = u_all[ix]
    W0 = np.sqrt(V**2 - U0**2)
    x = (np.arange(sz)-sz/2)*sampling/r
    xy = np.meshgrid(x,x)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    th = np.arctan2(xy[0],xy[1]) + th_offset
    win = np.where(r < 1)
    wout = np.where(r >= 1)
    the_mode = np.zeros( (sz,sz) )
    the_mode[win] = special.jn(unsigned_j,r[win]*U0)
    scale = special.jn(unsigned_j,U0)/special.kn(unsigned_j,W0)
    the_mode[wout] = scale * special.kn(unsigned_j,r[wout]*W0)
    return the_mode/np.sqrt(np.sum(the_mode**2))*np.exp(1j*unsigned_j*th)

def compute_v_number(wavelength_in_mm, core_radius, numerical_aperture):
    """Computes the V number (can be interpreted as a kind of normalized optical frequency) for an optical fibre
    
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength of light in mm
    core_radius: float
        The core radius of the fibre in mm
    numerical_aperture: float
        The numerical aperture of the optical fibre, defined be refractive indices of the core and cladding
        
    Returns
    -------
    v: float
        The v number of the fibre
        
    """
    v = 2 * np.pi / wavelength_in_mm * core_radius * numerical_aperture
    return v

def apply_and_scale_turbulent_ef(turbulence, npix, wavelength, dx, seeing):
    """ Applies an atmosphere in the form of Kolmogorov turbulence to an initial wavefront and scales
    
    Parameters
    ----------
    npix: integer
        The size of the square of Kolmogorov turbulence generated
    wavelength: float
        The wavelength in mm. Amount of atmospheric distortion depends on the wavelength. 
    dx: float
        Resolution in mm/pixel
    seeing: float
        Seeing in arcseconds before magnification
    Returns
    -------
    turbulent_ef or 1.0: np.array([[...]...]) or 1.0
        Return an array of phase shifts in imperfect seeing, otherwise return 1.0, indicating no change to the incident wave.
    """
    if seeing > 0.0:
        # Convert seeing to radians
        seeing_in_radians = np.radians(seeing/3600.)
        
        # Generate the Kolmogorov turbulence
        #turbulence = optics_tools.kmf(npix)
        
        # Calculate r0 (Fried's parameter), which is a measure of the strength of seeing distortions
        r0 = 0.98 * wavelength / seeing_in_radians 
        
        # Apply the atmosphere and scale
        wf_in_radians = turbulence * np.sqrt(6.88*(dx/r0)**(5.0/3.0))
            
        # Convert the wavefront to an electric field
        turbulent_ef = np.exp(1.0j * wf_in_radians)
        
        return turbulent_ef
    else:
        # Do not apply phase distortions --> multiply by unity
        return 1.0

def calculate_fibre_mode(wavelength_in_mm, fibre_core_radius, numerical_aperture, npix, dx):
    """Computes the mode of the optical fibre.
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength in mm
    fibre_core_radius: float
        The radius of the fibre core in mm
    numerical_aperture: float
        The numerical aperture of the fibre
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel    
    Returns
    -------
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre
    """
    # Calculate the V number for the model
    v = compute_v_number(wavelength_in_mm, fibre_core_radius, numerical_aperture)
    
    # Use the V number to calculate the mode
    fibre_mode = mode_2d(v, fibre_core_radius, sampling=dx, sz=npix)

    return fibre_mode


def compute_coupling(npix, dx, electric_field, lens_width, fibre_mode, x_offset, y_offset):
    """Computes the coupling between the electric field and the optical fibre using an overlap integral.
    
    Parameters
    ----------
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel      
    electric_field: np.array([[...]...])
        The electric field at the fibre plane
    lens_width: float
        The width of the a single microlens (used for minimising the unnecessary calculations)
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre   
    x_offset: int
        x offset of the focal point at the fibre plane relative to the centre of the microlens.
    y_offset: int
        y offset of the focal point at the fibre plane relative to the centre of the microlens.           
   
    Returns
    -------
    coupling: float
        The coupling between the fibre mode and the electric_field (Max 1)
    """
    npix = int(npix)
    
    # Crop the electric field to the central 1/4
    low = npix//2 - int(lens_width / dx / 2) #* 3/8
    upper = npix//2 + int(lens_width / dx / 2) #* 5/8
    
    # Compute the fibre mode and shift (if required)
    fibre_mode = fibre_mode[(low + x_offset):(upper + x_offset), (low + y_offset):(upper + y_offset)]
    
    # Compute overlap integral - denominator first
    den = np.sum(np.abs(fibre_mode)**2) * np.sum(np.abs(electric_field)**2)
    
    #Crop the electric field and compute the numerator
    #electric_field = electric_field[low:upper,low:upper]
    num = np.abs(np.sum(fibre_mode*np.conj(electric_field)))**2

    coupling = num / den
    
    return coupling

def nglass(l, glass='sio2'):
    """Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength 
    """
    try:
        nl = len(l)
    except:
        l = [l]
        nl=1
    l = np.array(l)
    if (glass == 'sio2'):
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3,1.35120631e-2,97.9340025])
    elif (glass == 'bk7'):
        B = np.array([1.03961212,0.231792344,1.01046945])
        C = np.array([6.00069867e-3,2.00179144e-2,1.03560653e2])
    elif (glass == 'nf2'):
        B = np.array( [1.39757037,1.59201403e-1,1.26865430])
        C = np.array( [9.95906143e-3,5.46931752e-2,1.19248346e2])
    elif (glass == 'nsf11'):
        B = np.array([1.73759695E+00,   3.13747346E-01, 1.89878101E+00])
        C = np.array([1.31887070E-02,   6.23068142E-02, 1.55236290E+02])
    elif (glass == 'ncaf2'):
        B = np.array([0.5675888, 0.4710914, 3.8484723])
        C = np.array([0.050263605,  0.1003909,  34.649040])**2
    elif (glass == 'mgf2'):
        B = np.array([0.48755108,0.39875031,2.3120353])
        C = np.array([0.04338408,0.09461442,23.793604])**2
    elif (glass == 'npk52a'):
        B = np.array([1.02960700E+00,1.88050600E-01,7.36488165E-01])
        C = np.array([5.16800155E-03,1.66658798E-02,1.38964129E+02])
    elif (glass == 'psf67'):
        B = np.array([1.97464225E+00,4.67095921E-01,2.43154209E+00])
        C = np.array([1.45772324E-02,6.69790359E-02,1.57444895E+02])
    elif (glass == 'npk51'):
        B = np.array([1.15610775E+00,1.53229344E-01,7.85618966E-01])
        C = np.array([5.85597402E-03,1.94072416E-02,1.40537046E+02])
    elif (glass == 'nfk51a'):
        B = np.array([9.71247817E-01,2.16901417E-01,9.04651666E-01])
        C = np.array([4.72301995E-03,1.53575612E-02,1.68681330E+02])
    elif (glass == 'si'): #https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
        B = np.array([10.6684293,0.0030434748,1.54133408])
        C = np.array([0.301516485,1.13475115,1104])**2
    #elif (glass == 'zns'): #https://refractiveindex.info/?shelf=main&book=ZnS&page=Debenham
    #    B = np.array([7.393, 0.14383, 4430.99])
    #    C = np.array([0, 0.2421, 36.71])**2
    elif (glass == 'znse'): #https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
        B = np.array([4.45813734,0.467216334,2.89566290])
        C = np.array([0.200859853,0.391371166,47.1362108])**2
    elif (glass == 'noa61'):
        n = 1.5375 + 8290.45/(l*1000)**2 - 2.11046/(l*1000)**4
        return n
    elif (glass == 'su8'):
        n = 1.5525 + 0.00629/l**2 + 0.0004/l**4
        return n
    elif (glass == 'epocore'):
        n = 1.572 + 0.0076/l**2 + 0.00046/l**4
        return n
    elif (glass == 'epoclad'):
        n = 1.560 + 0.0073/l**2 + 0.00038/l**4
        return n
    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
            n += B[i]*l**2/(l**2 - C[i])
    return np.sqrt(n)

#The following is directly from refractiveindex.info, and copied here because of
#UTF-8 encoding that doesn't seem to work with my python 2.7 installation.
#Author: Mikhail Polyanskiy
#(Ciddor 1996, https://doi.org/10.1364/AO.35.001566)

def Z(T,p,xw): #compressibility
    t=T-273.15
    a0 = 1.58123e-6   #K.Pa^-1
    a1 = -2.9331e-8   #Pa^-1
    a2 = 1.1043e-10   #K^-1.Pa^-1
    b0 = 5.707e-6     #K.Pa^-1
    b1 = -2.051e-8    #Pa^-1
    c0 = 1.9898e-4    #K.Pa^-1
    c1 = -2.376e-6    #Pa^-1
    d  = 1.83e-11     #K^2.Pa^-2
    e  = -0.765e-8    #K^2.Pa^-2
    return 1-(p/T)*(a0+a1*t+a2*t**2+(b0+b1*t)*xw+(c0+c1*t)*xw**2) + (p/T)**2*(d+e*xw**2)


def nm1_air(wave,t,p,h,xc):
    # wave: wavelength, 0.3 to 1.69 mu m 
    # t: temperature, -40 to +100 deg C
    # p: pressure, 80000 to 120000 Pa
    # h: fractional humidity, 0 to 1
    # xc: CO2 concentration, 0 to 2000 ppm

    sigma = 1/wave           #mu m^-1
    
    T= t + 273.15     #Temperature deg C -> K
    
    R = 8.314510      #gas constant, J/(mol.K)
    
    k0 = 238.0185     #mu m^-2
    k1 = 5792105      #mu m^-2
    k2 = 57.362       #mu m^-2
    k3 = 167917       #mu m^-2
 
    w0 = 295.235      #mu m^-2
    w1 = 2.6422       #mu m^-2
    w2 = -0.032380    #mu m^-4
    w3 = 0.004028     #mu m^-6
    
    A = 1.2378847e-5  #K^-2
    B = -1.9121316e-2 #K^-1
    C = 33.93711047
    D = -6.3431645e3  #K
    
    alpha = 1.00062
    beta = 3.14e-8       #Pa^-1,
    gamma = 5.6e-7        #deg C^-2

    #saturation vapor pressure of water vapor in air at temperature T
    if(t>=0):
        svp = np.exp(A*T**2 + B*T + C + D/T) #Pa
    else:
        svp = 10**(-2663.5/T+12.537)
    
    #enhancement factor of water vapor in air
    f = alpha + beta*p + gamma*t**2
    
    #molar fraction of water vapor in moist air
    xw = f*h*svp/p
    
    #refractive index of standard air at 15 deg C, 101325 Pa, 0% humidity, 450 ppm CO2
    nas = 1 + (k1/(k0-sigma**2)+k3/(k2-sigma**2))*1e-8
    
    #refractive index of standard air at 15 deg C, 101325 Pa, 0% humidity, xc ppm CO2
    naxs = 1 + (nas-1) * (1+0.534e-6*(xc-450))
    
    #refractive index of water vapor at standard conditions (20 deg C, 1333 Pa)
    nws = 1 + 1.022*(w0+w1*sigma**2+w2*sigma**4+w3*sigma**6)*1e-8
    
    Ma = 1e-3*(28.9635 + 12.011e-6*(xc-400)) #molar mass of dry air, kg/mol
    Mw = 0.018015                            #molar mass of water vapor, kg/mol
    
    Za = Z(288.15, 101325, 0)                #compressibility of dry air
    Zw = Z(293.15, 1333, 1)                  #compressibility of pure water vapor
    
    #Eq.4 with (T,P,xw) = (288.15, 101325, 0)
    rhoaxs = 101325*Ma/(Za*R*288.15)           #density of standard air
    
    #Eq 4 with (T,P,xw) = (293.15, 1333, 1)
    rhows  = 1333*Mw/(Zw*R*293.15)             #density of standard water vapor
    
    # two parts of Eq.4: rho=rhoa+rhow
    rhoa   = p*Ma/(Z(T,p,xw)*R*T)*(1-xw)       #density of the dry component of the moist air    
    rhow   = p*Mw/(Z(T,p,xw)*R*T)*xw           #density of the water vapor component
    
    nprop = (rhoa/rhoaxs)*(naxs-1) + (rhow/rhows)*(nws-1)
    
    return nprop
    

# model
def nm1_L_air(wave,t,p,h):
    # wave: wavelength, 2.8 to 4.2 mu m 
    # t: temperature, -40 to +100 deg C
    # p: pressure, 80000 to 120000 Pa
    # h: fractional humidity, 0 to 1
    H = 100*h
    T = t + 273.15
    #Constants
    # model parameters
    cref = [ 0.200049e-3,  0.145221e-9,   0.250951e-12, -0.745834e-15, -0.161432e-17,  0.352780e-20] # cm^j
    cT   = [ 0.588432e-1, -0.825182e-7,   0.137982e-9,   0.352420e-13, -0.730651e-15, -0.167911e-18] # cm^j · K
    cTT  = [-3.13579,      0.694124e-3,  -0.500604e-6,  -0.116668e-8,   0.209644e-11,  0.591037e-14] # cm^j · K^2
    cH   = [-0.108142e-7,  0.230102e-11, -0.154652e-14, -0.323014e-17,  0.630616e-20,  0.173880e-22] # cm^j · %^-1
    cHH  = [ 0.586812e-12, 0.312198e-16, -0.197792e-19, -0.461945e-22,  0.788398e-25,  0.245580e-27] # cm^j · %^-2
    cp   = [ 0.266900e-8,  0.168162e-14,  0.353075e-17, -0.963455e-20, -0.223079e-22,  0.453166e-25] # cm^j · Pa^-1
    cpp  = [ 0.608860e-17, 0.461560e-22,  0.184282e-24, -0.524471e-27, -0.121299e-29,  0.246512e-32] # cm^j · Pa^-2
    cTH  = [ 0.517962e-4, -0.112149e-7,   0.776507e-11,  0.172569e-13, -0.320582e-16, -0.899435e-19] # cm^j · K · %^-1
    cTp  = [ 0.778638e-6,  0.446396e-12,  0.784600e-15, -0.195151e-17, -0.542083e-20,  0.103530e-22] # cm^j · K · Pa^-1
    cHp  = [-0.217243e-15, 0.104747e-20, -0.523689e-23,  0.817386e-26,  0.309913e-28, -0.363491e-31] # cm^j · %^-1 · Pa^-1
    sigref = 1e4/3.4     # cm^−1
    Tref = 273.15+17.5 # K
    pref = 75000       # Pa
    Href = 10          #%

    sig = 1e4/wave # cm^-1
    nm1 = 0
    for j in range(0, 6):
        nm1 += ( cref[j] + cT[j]*(1/T-1/Tref) + cTT[j]*(1/T-1/Tref)**2
            + cH[j]*(H-Href) + cHH[j]*(H-Href)**2
            + cp[j]*(p-pref) + cpp[j]*(p-pref)**2
            + cTH[j]*(1/T-1/Tref)*(H-Href)
            + cTp[j]*(1/T-1/Tref)*(p-pref)
            + cHp[j]*(H-Href)*(p-pref) ) * (sig-sigref)**j   
    return nm1
        
def fresnel_reflection(n1, n2, theta=0):
    """
    Parameters
    ----------
    theta: float
        incidence angle in degrees
    
    Returns 
    -------
    Rp: float
        s (perpendicular) plane reflection
    Rs: float
        p (parallel) plane reflection
    """
    th = np.radians(theta)
    sqrt_term = np.sqrt(1-(n1/n2*np.sin(th))**2)
    Rs = (n1*np.cos(th) - n2*sqrt_term)**2/(n1*np.cos(th) + n2*sqrt_term)**2
    Rp = (n1*sqrt_term - n2*np.cos(th))**2/(n1*sqrt_term + n2*np.cos(th))**2
    return Rs, Rp
