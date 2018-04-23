import numpy as np

def slice(logp, x_init, width=1.0,max_steps=1000):
    """ Takes in a vector """
    x_init = np.array(x_init).astype('float64').ravel()
    #np.random.shuffle(x_init)
    dim = len(x_init)
    samples,x,xp,L,R = [], x_init.copy(), x_init.copy(), x_init.copy(), x_init.copy()

    for d in range(dim):
        yprime = logp(x)+np.log(np.random.uniform())
        u = np.random.uniform()
        L[d] = x[d] - width*u
        R[d] = x[d] + width*(1-u)

        #print(yprime,L,R)
        # Step out section 
        it=0
        while yprime < logp(L) and it<max_steps:
            L[d] = L[d]-width
            it+=1
        #print('on to the right',L[d])
        it = 0
        while yprime < logp(R) and it<max_steps:
            R[d] = R[d]+width
            it+=1
        #print('enter final:',np.random.uniform(L[d],R[d]))
        #print(type(xp[0]))
        while True:
            xp[d] = np.random.uniform(L[d],R[d])
            #print(xp[d],np.random.uniform(L[d],R[d]))
            #print(yprime,logp(xp))
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
        #print(xp)
        x[d] = xp[d]
        L[d] = xp[d]
        R[d] = xp[d]
        
    return x
