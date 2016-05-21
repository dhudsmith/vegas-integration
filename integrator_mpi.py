import sys
import vegas
import numpy as np
import integrand_mpi
import time

def domain(cutoff):
    return [[0, cutoff],    # kcm
            [0, cutoff],    # K
            [0, cutoff],    # Kp
            [0, np.pi / 2], # alpha
            [0, np.pi / 2], # alphap
            [0, np.pi],     # cosb
            [0, np.pi],     # cosg
            [0, 2*np.pi],    # phi
            [0, np.pi],     # cosbp
            [0, np.pi],     # cosgp
            [0, 2*np.pi]]   # phip

def speed_test(cutoff, niters, nevals, nbatch):

    dom = domain(cutoff)
    integ = vegas.Integrator(dom, nhcube_batch=nbatch)
    f = vegas.MPIintegrand(integrand_mpi.g)



    result = integ(f, nitn=niters, neval=nevals)

    if f.rank==0:
        print(result.summary())
        print('result = %s   Q = %.2f' % (result, result.Q))
        t1 = time.time()
        print('Cython time with MPI and vectorization:', t1 - t0)


def print_int(func, cutoff, niters, nevals, throwaway=10):

    fmpi = vegas.MPIintegrand(func)

    dom = domain(cutoff)
    integ = vegas.Integrator(dom, nhcube_batch=2000)

    t0 = time.time()

    integ(fmpi, nitn = throwaway, neval = nevals)

    result =  integ(fmpi, nitn=niters, neval=nevals)

    if fmpi.rank==0:
        print(result.summary())
        print('result = %s   Q = %.2f' % (result, result.Q))
        t1 = time.time()
        print('Time to calculate integral of %s:' % func.__name__, t1 - t0)


if __name__=="__main__":
    cutoff = float(sys.argv[1])
    niter = int(sys.argv[2])
    nevals = int(sys.argv[3])
    throwaway = int(sys.argv[4])

    res_g = print_int(integrand_mpi.g, cutoff, niter, nevals, throwaway)
    res_gK = print_int(integrand_mpi.gK, cutoff, niter, nevals, throwaway)
    res_gKp = print_int(integrand_mpi.gKp, cutoff, niter, nevals, throwaway)





