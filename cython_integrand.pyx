cimport vegas
import vegas as veg
from libc.math cimport sqrt, sin, cos
from cython_gsl cimport gsl_sf_bessel_Jn as jn
import numpy

cdef double _sqrt3o2 = sqrt(3./2.)
cdef double _sqrt2 = sqrt(2.)
cdef double _1by3 = 1./3.
cdef double _2by3 = 2./3.
cdef double _4by3 = 4./3.
cdef double _1by9 = 1./9.
cdef double _4by9 = 4./9.

def integrand(x):
    # Read in the input
    cdef double kcm     =   x[0]
    cdef double K       =   x[1]
    cdef double Kp      =   x[2]
    cdef double alpha   =   x[3]
    cdef double alphap  =   x[4]
    cdef double beta    =   x[5]
    cdef double gamma   =   x[6]
    cdef double phi     =   x[7]
    cdef double betap   =   x[8]
    cdef double gammap  =   x[9]
    cdef double phip    =   x[10]

    #For reuse
    cdef double cosb = cos(beta)
    cdef double sinb = sin(beta)
    cdef double cosg = cos(gamma)
    cdef double sing = sin(gamma)
    cdef double cosph = cos(phi)
    cdef double cosbp = cos(betap)
    cdef double sinbp = sin(betap)
    cdef double cosgp = cos(gammap)
    cdef double singp = sin(gammap)
    cdef double cosphp = cos(phip)


    # k312 and k12 definitions in terms of
    # integration variables
    cdef double k312    = _sqrt3o2 * K * cos(alpha)
    cdef double k312p   = _sqrt3o2 * Kp * cos(alphap)
    cdef double k12     = _sqrt2 * K * sin(alpha)
    cdef double k12p    = _sqrt2 * Kp * sin(alphap)

    # k1, k2, etc. variables in terms of integration
    # variables
       # k1, k2, etc. variables in terms of integration
    # variables
    cdef double k1s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         - _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         + kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    cdef double k2s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         + _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         - kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    cdef double k3s = kcm ** 2 \
         + _4by9 * k312 ** 2 \
         + _4by3 * kcm * k312 *cosb

    cdef double k1sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          - _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          + kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    cdef double k2sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          + _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          - kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    cdef double k3sp = kcm ** 2 \
         + _4by9 * k312p ** 2 \
         + _4by3 * kcm * k312p * cosbp

    # The actual integrand
    return kcm**2 * K**3 * Kp**3 * sin(2*alpha)**2 * sin(2*alphap)**2 \
            * sinb * sing * sinbp * singp \
            * jn(2,sqrt(k1s))/k1s * jn(2,sqrt(k2s))/k2s * jn(2,sqrt(k3s))/k3s \
            * jn(2,sqrt(k1sp))/k1sp * jn(2,sqrt(k2sp))/k2sp * jn(2,sqrt(k3sp))/k3sp

cdef double cintegrand(x):
    # Read in the input
    cdef double kcm     =   x[0]
    cdef double K       =   x[1]
    cdef double Kp      =   x[2]
    cdef double alpha   =   x[3]
    cdef double alphap  =   x[4]
    cdef double beta    =   x[5]
    cdef double gamma   =   x[6]
    cdef double phi     =   x[7]
    cdef double betap   =   x[8]
    cdef double gammap  =   x[9]
    cdef double phip    =   x[10]

    #For reuse
    cdef double cosb = cos(beta)
    cdef double sinb = sin(beta)
    cdef double cosg = cos(gamma)
    cdef double sing = sin(gamma)
    cdef double cosph = cos(phi)
    cdef double cosbp = cos(betap)
    cdef double sinbp = sin(betap)
    cdef double cosgp = cos(gammap)
    cdef double singp = sin(gammap)
    cdef double cosphp = cos(phip)


    # k312 and k12 definitions in terms of
    # integration variables
    cdef double k312    = _sqrt3o2 * K * cos(alpha)
    cdef double k312p   = _sqrt3o2 * Kp * cos(alphap)
    cdef double k12     = _sqrt2 * K * sin(alpha)
    cdef double k12p    = _sqrt2 * Kp * sin(alphap)

    # k1, k2, etc. variables in terms of integration
    # variables
       # k1, k2, etc. variables in terms of integration
    # variables
    cdef double k1s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         - _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         + kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    cdef double k2s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         + _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         - kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    cdef double k3s = kcm ** 2 \
         + _4by9 * k312 ** 2 \
         + _4by3 * kcm * k312 *cosb

    cdef double k1sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          - _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          + kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    cdef double k2sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          + _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          - kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    cdef double k3sp = kcm ** 2 \
         + _4by9 * k312p ** 2 \
         + _4by3 * kcm * k312p * cosbp

    # The actual integrand
    return kcm**2 * K**3 * Kp**3 * sin(2*alpha)**2 * sin(2*alphap)**2 \
            * sinb * sing * sinbp * singp \
            * jn(2,sqrt(k1s))/k1s * jn(2,sqrt(k2s))/k2s * jn(2,sqrt(k3s))/k3s \
            * jn(2,sqrt(k1sp))/k1sp * jn(2,sqrt(k2sp))/k2sp * jn(2,sqrt(k3sp))/k3sp

cdef class cython_integrand(vegas.BatchIntegrand):
    def __call__(self, double[:, ::1] x):
        cdef double[::1] f = numpy.empty(x.shape[0], float)
        for i in range(f.shape[0]):
            f[i] = cintegrand(x[i,::1])
        return f

@veg.batchintegrand
def fbatch(x):
    cdef double[::1] f
    f = numpy.empty(x.shape[0], float)
    for i in range(f.shape[0]):
        f[i] = cintegrand(x[i,::1])
    return f

def fMPI(double[:,::1] x):
    cdef double[::1] f
    f = numpy.empty(x.shape[0], float)
    for i in range(f.shape[0]):
        f[i] = cintegrand(x[i,::1])
    return f


def main():
    import time

    num = 10**3

    x = [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    y = numpy.array([x]*num, float)

    t0 = time.time()
    for i in range(0,num):
        integrand(x)
    t1 = time.time()
    print("Loop:", t1-t0)

    t0 = time.time()
    f = cython_integrand()
    #print(numpy.asarray(f(y)))
    f(y)
    t1 = time.time()
    print("Vector:", t1-t0)

    print(integrand(x))