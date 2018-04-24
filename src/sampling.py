import numpy as np

def slice(logp, x_init, width=1.0,max_steps=1000):
    """ Slice sampler for integration over the acquisition functions. This sampler performs one 
    "slice" and returns the valid sample from the distribution.  The purpose of performing 
    the one slice is sampling and then computing the improvement metric and then sampling again.
    -------
    Args: logp; function, that takes in a vector of shape x_init and returns a probability.
          x_init; numpy array; should be only 1d
          width; float, the size of the step when we step out
          max_steps; int, the maximum number of steps before we quit and just sample 
    -------
    Returns: numpy array that is a valid sample from the probability distribution governed by logp
    """
    x_init = np.array(x_init).astype('float64').ravel()
    dim = len(x_init)
    samples,x,xp,L,R = [], x_init.copy(), x_init.copy(), x_init.copy(), x_init.copy()

    for d in range(dim):
        yprime = logp(x)+np.log(np.random.uniform())
        u = np.random.uniform()
        L[d] = x[d] - width*u
        R[d] = x[d] + width*(1-u)

        # Step out section 
        it=0
        while yprime < logp(L) and it<max_steps:
            L[d] = L[d]-width
            it+=1
        it = 0
        while yprime < logp(R) and it<max_steps:
            R[d] = R[d]+width
            it+=1

        while True:
            xp[d] = np.random.uniform(L[d],R[d])
            if yprime < logp(xp):
                break
            else:
                # Shrink in section
                if xp[d] > x[d]:
                    R[d] = xp[d]
                elif xp[d] < x[d]:
                    L[d] = xp[d]
                else: 
                    raise ValueError('Step has shrunk to much and cannot continue')

        x[d] = xp[d]
        L[d] = xp[d]
        R[d] = xp[d]
        
    return x
