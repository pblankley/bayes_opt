import numpy as np

def slice(logp, nsamps, x_init, width=1.0):
    """ Takes in a vector """
    x_init = np.array(x_init).ravel()
    dim = len(x_init)
    samples,x,xp,L,R = [], x_init.copy(), x_init.copy(), x_init.copy(), x_init.copy()

    for d in range(dim):
        yprime = logp(x)+np.log(np.random.uniform())
        u = np.random.uniform()
        L[d] = x[d] - width*u
        R[d] = x[d] + width*(1-u)

        # Step out section 
        while yprime < logp(L):
            L[d] = L[d]-width
        while yprime < logp(R):
            R[d] = R[d]+width
        
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
