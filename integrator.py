import vegas
import numpy as np
import integrand
import cython_integrand
import time

f = integrand.integrand
fc = cython_integrand.integrand
fcbatch = cython_integrand.fbatch
fmpi = vegas.MPIintegrand(cython_integrand.fMPI)

def domain(cutoff):
    pi = np.pi
    return [[0, cutoff],    # kcm
            [0, cutoff],    # K
            [0, cutoff],    # Kp
            [0, pi / 2], # alpha
            [0, pi / 2], # alphap
            [0, pi],     # beta
            [0, pi],     # gamma
            [0, 2*pi],    # phi
            [0, pi],     # cosbp
            [0, pi],     # cosgp
            [0, 2*pi]]   # phip

def speed_test(cutoff, niters, nevals, nbatch):
    ##################################exi
    # setup
    ##################################
    dom = domain(cutoff)

    ##################################
    # integrand.py
    ##################################
    t0 = time.time()
    integ = vegas.Integrator(dom)

    result = integ(f, nitn=niters, neval=nevals)
    print(result.summary())
    print('result = %s   Q = %.2f' % (result, result.Q))
    t1 = time.time()
    print('Python time:', t1 - t0)

    ##################################
    # cython_integrand.integrand
    ##################################
    t0 = time.time()

    integ = vegas.Integrator(dom)

    result = integ(fc, nitn=niters, neval=nevals)
    print(result.summary())
    print('result = %s   Q = %.2f' % (result, result.Q))

    t1 = time.time()
    print('Cython time without vectorization:', t1 - t0)

    ##################################
    # cython_integrand.cython_integrand
    ##################################
    t0 = time.time()

    integ = vegas.Integrator(dom, nhcube_batch=nbatch)

    result = integ(fcbatch, nitn=niters, neval=nevals)
    print(result.summary())
    print('result = %s   Q = %.2f' % (result, result.Q))

    t1 = time.time()
    print('Cython time with vectorization:', t1 - t0)

    ##################################
    # cython_integrand.cython_integrand
    ##################################
    t0 = time.time()

    integ = vegas.Integrator(dom, nhcube_batch=nbatch)

    result = integ(fmpi, nitn=niters, neval=nevals)
    print(result.summary())
    print('result = %s   Q = %.2f' % (result, result.Q))

    t1 = time.time()
    print('Cython time with MPI and vectorization:', t1 - t0)

def cutoff_test(list_cutoffs, niters, nevals):

    results = []

    for L in list_cutoffs:
        # Initialize the integrator
        integ = vegas.Integrator(domain(L))

        # Iterate to get a good estimate of the density before printing results
        integ(fc, nitn=10, neval=nevals)

        # Print the results
        result = integ(fc2, nitn=niters, neval=nevals)
        print('cutoff = %i   result = %s   Q = %.2f' % (L, result, result.Q))

        # Record the results
        results.append([L, result.mean, result.sdev, result.Q])

    results = np.array(results)
    with open("cutoff_dependence.txt", 'wb') as file:
        np.savetxt(file, results, fmt='%i %0.8f %0.8f %0.2f', delimiter=',',
                   header= 'Dependence of L3 integral on momentum cutoff\n'
                           'niters =' +str(niters)+'  nevals = '+str(nevals) +
                           '\n' + 'cutoff  integral  stat.err.  Q')

def evaluate_integral(cutoff, niters, nevals, throwaway=10):
    dom = domain(cutoff)
    integ = vegas.Integrator(dom)
    integ(fc, nitn = throwaway, neval = nevals)
    return(integ(fc, nitn=niters, neval=nevals))

if __name__=="__main__":
    speed_test(16, 15, 2*10**3,2000)




