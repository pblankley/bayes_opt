import numpy as np
from scipy.stats import norm
import scipy.linalg as spla
from tqdm import tqdm
import sampling 

class GP(object):
    """ Gaussian process object 
    ---------
    INIT VALUES:
    ---------
    x; numpy array of shape n x m with n points and m features. 
    y; numpy array of shape n or n x 1 with n points 
    kernel; function, a kernel function of type Matern52 or SquaredExp
    noise; initial noise to add to the draws from the posterior
    
    =========
    CALCULATED VALUES
    ---------
    there are 3+m where m is the number of features hyperparameters to a GP 
    the first three are as follows:
    mean; initialized by the mean of the response, y
    amplitude (amp); initialized by the std of the response, y
    noise; initialized to 1e-3 or user specification
    thetas; m dimensional numpy array of ones 
    ---------
    """
    def __init__(self, x, y, kernel, noise=1e-3):
        self.obsx = np.array(x)
        self.obsy = np.array(y).reshape(-1,1)
        self.mean = np.mean(y)
        self.amp = np.std(y)
        self.kernel = kernel
        self.dim = x.shape[1]
        self.thetas = np.ones(self.dim)
        self.noise = noise
        
    def set_params(self,**kwargs):
        """ Sets arguments for the GP object. Options are: {x,y, kernel, thetas, dim, means} """
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def cov(self,x,xp,thetas=None): 
        """ Get the covariance matrix for the GP based on observed data
        -------
        Args: x; numpy array of shape nxm 
              xp, numpy array of shape pxm 
              thetas; defaults to array of ones but can be any length m vector 
        -------
        Returns: numpy array of shape nxp 
        """
        return self.amp*self.kernel(x,xp,thetas)
    
    def draw_prior(self,ndraws):
        """ Draw from the prior over the functions of the GP. Math in draw_posterior
        -------
        Args: ndraws; int, the number of draws from the prior 
        -------
        Returns: numpy array of shape 'length of initial x' X ndraws 
        """
        dim = self.obsx.shape[0]
        # The diagonal addition below is for numeric stability
        cov_mat = self.cov(self.obsx,self.obsx,self.thetas)+np.eye(dim)*1e-7
        L_chol = np.linalg.cholesky(cov_mat)
        return L_chol @ np.random.normal(size=(dim, ndraws))

    def draw_posterior(self,xp):
        """ The math is given as follows:
        
        Covaraince matrix can be broken into three sections 
        
        | K   K* |
        | K*T K**|
        
        where K = kernel(x,x), K* = kernel(x,x*), K** = kernel(x*,x*)
        
        In math the mean of the posterior is given by: 
            mu = K*T @ K^-1 @ y where dimensions are (MxN), (NXN), (Nx1) result: (Mx1)
        
        We use a cholesky decomp so we dont (stupidly) take the inverse of K 
        Since K = LL^T with cholesky: K*T @ L^-1@L^-1^T @ y
        We can set up K*T @ L^-1 and L^-1^T @ y as lower tri eq's to solve, giving:
            mu = solve(L,K*T).T @ solve(L, y) 
        
        In math the covariance of the posterior is given by: 
            cov = K** - K*T @ K^-1 @ K* where dimensions are (MxM), (MxN), (NxN) (NxM) result: (MxM)
        
        The covariance can be solved for with similar numerical tricks: 
        Since K = LL^T with cholesky: K** -  K*T @ L^-1@L^-1^T @ K*
            cov = K** -  solve(L,K*T).T @ solve(L, K*T)
        ---------
        Args: xp; numpy array of shape nxm when n is an arb number of points and m must be the same 
                    number of features as the original input x 
        ---------
        Returns; tuple of a mean vector and a std vector  of length n when xp is size nxm
        ---------
        Note: the n here does NOT have to be the same as the original input x array 

        Source: http://www.cs.ubc.ca/~nando/540-2013/lectures/l6.pdf
        """
        dim1, dim2 = self.obsx.shape[0],xp.shape[0]
        
        x_diff = self.obsy - np.mean(self.obsx)
        
        # Get covariance matrix sections
        cov = self.cov(self.obsx,self.obsx,self.thetas) + np.eye(dim1)*1e-10 + np.eye(dim1)*self.noise
        cov_star = self.cov(self.obsx,xp,self.thetas) + np.eye(dim1,dim2)*1e-10 + np.eye(dim1,dim2)*self.noise
        cov_ss = self.cov(xp,xp,self.thetas) + np.eye(dim2)*1e-10 + np.eye(dim2)*self.noise
        
        # Get useful cholsesky decompositions
        L = np.linalg.cholesky(cov)
        Linv_cov_s = np.linalg.solve(L,cov_star) # TODO: maybe change this to a lower triangular solve for efficiency? 
        
        # Calculate the posterior means and covariance matrix
        self.means = Linv_cov_s.T @ np.linalg.solve(L, self.obsy.reshape(-1,1))
        self.cmat = cov_ss - Linv_cov_s.T @ Linv_cov_s + np.eye(dim2)*1e-10
        self.std = np.sqrt(np.diag(cov_ss) - np.sum(Linv_cov_s**2, axis=0))
        
        return self.means.ravel(), self.std 
    
    def draw_posterior_pred(self,ndraws):
        """ Draw from the posterior predictive 
        -------
        Args: ndraws; int the number of draws
        -------
        Returns: a n x ndraws matrix when n is the number of points in the most recently used posterior call 
        -------
        Note: you must run the posterior function before this will work 
        """
        return self.means + self.cmat@np.random.normal(size=(self.cmat.shape[0],ndraws))

    def logp(self, x, y):
        """ This is the marginal log likelihood as I see it from the math and from the sklearn implementation.
        
        Math is as follows with notation from draw_posterior function: 
            NLL = -0.5*log(l1_norm(K))-0.5 * y^T @ K^-1 @ y - n/2*log(2pi)
        ------
        Args: x; numpy array of pxn shape 
              y; numpy array of px1 shape
        ------
        Returns; float, the log likelihood at x and y 
        ------
        Source: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/gaussian_process/gpr.py#L378 
        """
        mean  = np.mean(y)

        cov   = self.amp * (self.cov(x, x, self.thetas) + 1e-7*np.eye(x.shape[0])) + self.noise*np.eye(x.shape[0])
        chol  = np.linalg.cholesky(cov)
        solve = spla.cho_solve((chol, True), y-mean)
        lp = np.sum(-0.5*(y-mean).T@solve)- np.log(np.diag(chol)).sum()- cov.shape[0] / 2*np.log(2*np.pi)
        return lp

    def pi_metric(self,xp):
        """ Probability of improvement metric 
        ---------
        Args: xp; numpy array of shape nxm when n is an arb number of points and m must be the same 
                    number of features as the original input x 
        ---------
        Returns; vector of length n when xp is size nxm
        """
        best = np.min(self.obsy)
        means, stds = self.draw_posterior(xp)
        gamma = (best - means)/(stds+1e-10)
        return norm.cdf(gamma)

    def ei_metric(self,xp):
        """ Expected improvement metric
        ---------
        Args: xp; numpy array of shape nxm when n is an arb number of points and m must be the same 
                    number of features as the original input x 
        ---------
        Returns; vector of length n when xp is size nxm
        """
        best = np.min(self.obsy)
        means, stds = self.draw_posterior(xp)
        gamma = (best - means)/(stds+1e-10)
        return stds * (gamma*norm.cdf(gamma) + norm.pdf(gamma))
    
    def ucb_metric(self,xp,k):
        """ GP Upper Confidence Bound metric
        ---------
        Args: xp; numpy array of shape nxm when n is an arb number of points and m must be the same 
                    number of features as the original input x 
        ---------
        Returns; vector of length n when xp is size nxm
        """
        means, stds = self.draw_posterior(xp)
        return -means + k*stds

    def next_gp_hyper(self):
        """ Find the next best GP hyperparameters by integrating them out of the acquisition function
        --------
        Args: None
        --------
        Returns: None, but is internally sets the amp, mean, and thetas parameters for the GP 
        --------
        Note: we integrate over noise but we do not change its value, e.g. the noise coefficient is constant 
        """
        x,y = self.obsx, self.obsy

        def lp_params(params):
            """ The log probability w.r.t. the amp, mean, and noise of the GP  """
            amp, mean, noise = params

            if mean > np.max(y) or mean < np.min(y):
                return -np.inf

            if amp < 0 or noise < 0:
                return -np.inf

            cov   = amp * (self.cov(x, x, self.thetas) + 1e-6*np.eye(x.shape[0])) + noise*np.eye(x.shape[0])
            chol  = np.linalg.cholesky(cov)
            solve = spla.cho_solve((chol, True), y-mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot((y-self.mean).T, solve)
            return lp

        new_params = sampling.slice(lp_params, np.array([self.amp,self.mean,self.noise]))
        self.amp, self.mean, self.noise = new_params
        self.noise = 1e-3

        def lp_thetas(thetas):
            """ The log probability w.r.t. the thetas of the GP """
            if np.any(thetas < 0) or np.any(thetas > 2.0):
                return -np.inf
            cov   = self.amp * (self.cov(x, x, thetas) + 1e-6*np.eye(x.shape[0])) + self.noise*np.eye(x.shape[0])
            chol  = np.linalg.cholesky(cov)
            solve = spla.cho_solve((chol, True), y-self.mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot((y-self.mean).T, solve)
            return lp

        new_thetas = sampling.slice(lp_thetas,self.thetas)
        self.thetas = new_thetas

    def next_proposal(self,xp,metric='ei',mcmc_itr=10):
        """ Find the next best proposal value based on the metric specified 
        -------
        Args: xp; numpy array of shape nxm 
              metric; str, one of the following 
                    'ei' - expected improvement - default 
                    'pi' - probability of improvement 
                    'ucb' or 'lcb' - GP lower bound 
              mcmc_itr; int, the number of iterations we take from the slice sampler 
        -------
        Returns: int, the index value in xp of the best proposed value. 
        """
        metric_values = np.zeros((xp.shape[0], mcmc_itr))
        if metric not in {'ei','pi','ucb','lcb'}:
            raise ValueError(f'Metric type not implemented {metric}')

        for itr in range(mcmc_itr):
            self.next_gp_hyper()
            if metric=='ei':
                metric_values[:,itr] = self.ei_metric(xp)
            if metric=='pi':
                metric_values[:,itr] = self.pi_metric(xp)
            if metric=='ucb' or metric=='lcb':
                metric_values[:,itr] = self.ucb_metric(xp)

        best_cand = np.argmax(metric_values.mean(axis=1))
        return best_cand

    def find_best(self,func,xp,iters,metric='ei',mcmc_itr=10):
        """ For the specified number of iterations, find the best values that minimize the 
        function that is passed.
        -------
        Args: func; function to be minimized.
              xp; numpy array of shape nxm e.g. the hypercube of proposed points 
              iters; int, the number of iterations to perform before exiting.
              metric; str, the metric to use, defaults to 'ei'
              mcmc_itr; int, mcmc interations for slice sampling, defaults to 10 
        -------
        Returns: tuple with tuple of best value and best result and list of entire 
                    history of points tried
        """ 
        history = []
        for i in tqdm(range(iters)):
            next_prop = self.next_proposal(xp, metric, mcmc_itr)
            next_val = xp[next_prop].reshape(-1,xp.shape[1])
            prop_res = func(next_val).reshape(-1,1)
            self.obsx = np.append(self.obsx, next_val, axis=0)
            self.obsy = np.append(self.obsy, prop_res, axis=0)
            history.append((next_val,prop_res))
        return min(history,key=lambda x: x[1]), history


########################## KERNELS ###############################

def pdist2(x,xp,theta):
    """ Function to calculate the squared pairwise distances efficiently for the kernel 
    -------
    Args: x; numpy array of shape nxm 
          xp, numpy array of shape pxm 
          thetas; defaults to array of ones but can be any length m vector 
    -------
    Returns: numpy array of shape nxp 
    -------
    Source: Idea from Jasper's code for efficiency.  The naive calculation takes a lot longer
    """
    xm = x/theta
    xpm = xp/theta
    return np.abs(np.sum(xm*xm,axis=1).reshape(-1,1) - 2*xm@xpm.T  \
                    + np.sum(xpm*xpm,axis=1).reshape(-1,1).T)

def SquaredExp(x,xp,theta=None):
    """ Squared exponential kernel function  
    -------
    Args: x; numpy array of shape nxm 
          xp, numpy array of shape pxm 
          thetas; defaults to array of ones but can be any length m vector 
    -------
    Returns: numpy array of shape nxp 
    """
    if theta is None:
        theta = np.ones((1,x.shape[1]))
    pd2 = pdist2(x,xp,theta)
    return np.exp(-0.5*pd2)

def Matern52(x,xp,theta=None):
    """ Matern 52  kernel function  
    -------
    Args: x; numpy array of shape nxm 
          xp, numpy array of shape pxm 
          thetas; defaults to array of ones but can be any length m vector 
    -------
    Returns: numpy array of shape nxp 
    """
    if theta is None:
        theta = np.ones((1,x.shape[1]))
    pd2 = pdist2(x,xp,theta)
    return (1.0+np.sqrt(3.0*pd2)+5.0*pd2/3.0 )*np.exp(-np.sqrt(5.0*pd2))

##################################################################
