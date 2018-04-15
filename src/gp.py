import numpy as np
import scipy.linalg as spla

class GP(object):
    """ Gaussian process object """
    def __init__(self, x, y, kernel, noise=0.0):
        self.obsx = np.array(x)
        self.obsy = np.array(y)
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
        """ Get the covariance matrix for the GP based on observed data """
        return self.amp*self.kernel(x,xp,thetas)
    
    def draw_prior(self,ndraws):
        """ Drqw from the prior over the functions of the GP. Math in draw_posterior """
        dim = self.obsx.shape[0]
        # The diagonal addition below is for numeric stability
        cov_mat = self.cov(self.obsx,self.obsx,self.thetas)+np.eye(dim)*1e-7
        L_chol = np.linalg.cholesky(cov_mat)
        return L_chol @ np.random.normal(size=(dim, ndraws))

    def draw_posterior(self,xp):
        """ The math is given as follows:
        Q: Do we want to do a noisy version or not? 
        
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
        
        Source: http://www.cs.ubc.ca/~nando/540-2013/lectures/l6.pdf
        """
        dim1, dim2 = self.obsx.shape[0],xp.shape[0]
        
        x_diff = self.obsy - np.mean(self.obsx)
        
        # Get covariance matrix sections
        cov = self.cov(self.obsx,self.obsx,self.thetas)+np.eye(dim1)*1e-7+ np.eye(dim1)*self.noise
        cov_star = self.cov(self.obsx,xp,self.thetas)+np.eye(dim1,dim2)*1e-7+ np.eye(dim1,dim2)*self.noise
        cov_ss = self.cov(xp,xp,self.thetas)+np.eye(dim2)*1e-7+ np.eye(dim2)*self.noise
        
        # Get useful cholsesky decompositions
        L = np.linalg.cholesky(cov)
        Linv_cov_s = np.linalg.solve(L,cov_star)
        
        # Calculate the posterior means and covariance matrix
        self.means = Linv_cov_s.T @ np.linalg.solve(L, self.obsy.reshape(-1,1))
        self.cmat = cov_ss - Linv_cov_s.T @ Linv_cov_s +np.eye(dim2)*1e-7
        self.std = np.sqrt(np.diag(cov_ss) - np.sum(Linv_cov_s**2, axis=0))
        
        return self.means.ravel(), self.std 
    
    def draw_posterior_pred(self,ndraws):
        return self.means + self.cmat@np.random.normal(size=(self.cmat.shape[0],ndraws))

    def logprob(self, comp, vals):
        mean  = np.mean(vals)

        amp2  = self.amp
        noise = self.noise

        cov   = amp2 * (self.cov(comp,comp,self.thetas) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
        chol  = spla.cholesky(cov, lower=True)
        #return chol
        solve = spla.cho_solve((chol, True), vals - mean)
        #return solve 
        lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot((vals-mean).T, solve)
        return lp
    
    def logp(self, x, y):
        """ This is the marginal log likelihood as I see it from the math and from the sklearn implementation.
        
        Math is as follows with notation from draw_posterior function: 
            NLL = -0.5*log(l1_norm(K))-0.5 * y^T @ K^-1 @ y - n/2*log(2pi)

        We just use a cholskey decomp to get rid of the inverse and solve.  Simple, right? 
        - problem 1: chol_solve gives a FREAKING DIFFERENT RESULT than solve 
        - problem 2: this is quite different from Jasper's implementation, although not really a problem bc just a scalar factor 

        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/gaussian_process/gpr.py#L378 """
        mean  = np.mean(y) # could use for normalization if needed 

        cov   = self.amp * (self.cov(x, x, self.thetas) + 1e-7*np.eye(x.shape[0])) + self.noise*np.eye(x.shape[0])
        chol  = np.linalg.cholesky(cov)
        #return chol
        solve = spla.cho_solve((chol, True), y-mean)
        #solve = np.linalg.solve(chol, y-mean)
        #return solve
        lp = np.sum(-0.5*y.T@solve)- np.log(np.diag(chol)).sum()- cov.shape[0] / 2*np.log(2*np.pi)
        return lp

########################## KERNELS ###############################
def pdist2(x,xp,theta):
    # Idea from Jasper's code for efficiency.  The naive calculation takes a lot longer
    xm = x/theta
    xpm = xp/theta
    return np.abs(np.sum(xm*xm,axis=1).reshape(-1,1) - 2*xm@xpm.T  \
                    + np.sum(xpm*xpm,axis=1).reshape(-1,1).T)

def SquaredExp(x,xp,theta=None):
    if theta is None:
        theta = np.ones((1,x.shape[1]))
    pd2 = pdist2(x,xp,theta)
    return np.exp(-0.5*pd2)

def Matern52(x,xp,theta=None):
    if theta is None:
        theta = np.ones((1,x.shape[1]))
    pd2 = pdist2(x,xp,theta)
    return (1.0+np.sqrt(3.0*pd2)+5.0*pd2/3.0 )*np.exp(-np.sqrt(5.0*pd2))

##################################################################
