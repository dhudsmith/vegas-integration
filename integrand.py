from math import sin, cos, sqrt
from scipy.special import jn

_sqrt3o2 = sqrt(3./2.)
_sqrt2 = sqrt(2.)
_1by3 = 1./3.
_2by3 = 2./3.
_4by3 = 4./3.
_1by9 = 1./9.
_4by9 = 4./9.


def integrand(x):
    # Read in the input
    kcm     =   x[0]
    K       =   x[1]
    Kp      =   x[2]
    alpha   =   x[3]
    alphap  =   x[4]
    beta    =   x[5]
    gamma    =  x[6]
    phi     =   x[7]
    betap   =   x[8]
    gammap   =  x[9]
    phip   =    x[10]

    #For reuse
    cosb = cos(beta)
    sinb = sin(beta)
    cosg = cos(gamma)
    sing = sin(gamma)
    cosph = cos(phi)
    cosbp = cos(betap)
    sinbp = sin(betap)
    cosgp = cos(gammap)
    singp = sin(gammap)
    cosphp = cos(phip)


    # k312 and k12 definitions in terms of
    # integration variables
    k312    = _sqrt3o2 * K * cos(alpha)
    k312p   = _sqrt3o2 * Kp * cos(alphap)
    k12     = _sqrt2 * K * sin(alpha)
    k12p    = _sqrt2 * Kp * sin(alphap)

    # k1, k2, etc. variables in terms of integration
    # variables
    k1s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         - _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         + kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    k2s = kcm**2 \
         + _1by9 * k312**2 \
         + 0.25 * k12**2 \
         + _1by3 * k12 * k312 * (cosb*cosg + sinb*sing*cosph) \
         - kcm * k12 * cosg \
         - _2by3 * k312 * kcm * cosb
    k3s = kcm ** 2 \
         + _4by9 * k312 ** 2 \
         + _4by3 * kcm * k312 *cosb

    k1sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          - _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          + kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    k2sp = kcm ** 2 \
          + _1by9 * k312p ** 2 \
          + 0.25 * k12p ** 2 \
          + _1by3 * k12p * k312p * (cosbp * cosgp + sinbp * singp * cosphp) \
          - kcm * k12p * cosgp \
          - _2by3 * k312p * kcm * cosbp
    k3sp = kcm ** 2 \
         + _4by9 * k312p ** 2 \
         + _4by3 * kcm * k312p * cosbp

    # The actual integrand
    return kcm**2 * K**3 * Kp**3 * sin(2*alpha)**2 * sin(2*alphap)**2 \
            * sinb * sing * sinbp * singp \
            * jn(2,sqrt(k1s))/k1s * jn(2,sqrt(k2s))/k2s * jn(2,sqrt(k3s))/k3s \
            * jn(2,sqrt(k1sp))/k1sp * jn(2,sqrt(k2sp))/k2sp * jn(2,sqrt(k3sp))/k3sp

if __name__=="__main__":
    import time
    x = [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

    t0 = time.time()
    for i in range(0,100000):
        integrand(x)
    t1 = time.time()

    print(integrand(x))

    print(t1-t0)