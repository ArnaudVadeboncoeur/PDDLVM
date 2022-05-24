
from .utilities import *
from .utilities import _PI

class PDE_State:
    
    def __init__(self, xFE, dimK=4, dimU=11, dimF=2):
        
        self.xFE   = xFE
        self.dimK  = dimK
        self.dimU  = dimU
        self.dimF  = dimF

        self.dimBl = 1
        self.dimBr = 1
       
        self.kappa   = tf.constant(0.1, shape=[self.dimK] , dtype=floatf)
        self.forcing = tf.constant(5. , shape= [self.dimF],  dtype=floatf)
        self.bdLeft  = tf.constant(0. , shape=[], dtype=floatf)
        self.bdRight = tf.constant(0.5, shape=[], dtype=floatf)
        
        self.soft_0_9e9 = tfb.SoftClip(low=tf.constant(1e-10,dtype=floatf), high=tf.constant(9e9,dtype=floatf))
        self.softplus = tfb.Softplus()

        self.kappaBij    = tfb.Softplus( )
        self.forcingBij  = tfb.Identity()
        self.uBij        = tfb.Identity()
        self.dirichiletBij = tfb.JointMap((tfb.Identity(), tfb.Identity()))
        
        self.TXFEMatU = self.TOnMesh( self.dimU ) #[dimU x points]
        self.TXFEMatF = self.TOnMesh( self.dimF ) #[dimU x points]
        self.elmCent  = tffloat(self.xFE[:-1] + np.diff(self.xFE))
        self.TCXFEMatK = self.genChebyT(self.dimK, self.elmCent )

        self.elmCent  = tffloat(self.xFE[:-1] + np.diff(self.xFE) / 2.)
        epsiCorr      = tffloat(tf.concat([[1e-1], tf.constant(0.,shape=[self.xFE.shape[0]-2]), [-1e-1]], 0) )
        print(self.xFE+epsiCorr)
        #self.wMesh    = 1./tfm.sqrt(1. - (self.xFE+epsiCorr)**2)
        self.wMesh    = 1./tfm.sqrt(1. - (self.xFE)**2)
        self.deltaXFE = tffloat(np.diff(self.xFE))

        # self.u_fun = getattr(self, 'chebyFunKO{}'.format(str(self.dimU-1)))

    '''Our Functions for doing Learning on the Mesh'''
    def forcing_fun(self, x):
        return tf.squeeze(self.forcingBij(self.genChebyFun( self.forcing[:,None], tf.reshape(x,[-1])) ))
    
    def forcing_funXFE(self):
        return tf.squeeze( self.forcingBij(tf.matmul(self.TXFEMatF, self.forcing[:,None], transpose_a=True) ))
 
    def u_funXFE(self, a):
        """
        Takes in  a as [numCoeffs x numFuncs].
        """
        return tf.squeeze(self.uBij(tf.matmul(self.TXFEMatU, a, transpose_a=True)))
    
    def kappa_fun(self, x):
        return tf.squeeze(self.kappaBij( self.genChebyFun( self.kappa[:,None], tf.reshape(x,[-1])) ))
    
    def kappa_funCXFE(self):
        return tf.squeeze(self.kappaBij( tf.matmul(self.TCXFEMatK, self.kappa[:,None], transpose_a=True) ))
 
    def dirichlet_fun( self ):
        return self.dirichiletBij((self.bdLeft, self.bdRight))


    '''Utilities for Sheby Funcs'''
    def chebyOnMesh(self, a):
        '''a is [dimCoeffs, numFunc]
        returns cheby of function a coeffs on mesh xFE as [f_1(x), f_2(x), ..,] 
        '''
        TMat = tf.vectorized_map( self.xFEChebyBind, tf.range(0, a.shape[0], dtype=floatf) )
        return tf.matmul(TMat, a, transpose_a=True)

    def genChebyFun(self, a, x):
        '''a is [dimCoeffs, numFunc], x is [numPoints]
        returns cheby of function a coeffs on given x as [f_1(x), f_2(x), ..,] 
        '''
        TMat = self.genChebyT(a.shape[0], x)
        return tf.matmul(TMat, a, transpose_a=True)
    
    def genChebyT(self, n ,x):
        '''n is numCoeffs, x is [numPoints]
        returns cheby T on location x
        '''
        def bind(n): return self.trigCheb(x, n)
        TMat = tf.vectorized_map( bind, tf.range(0, n, dtype=floatf) )
        return TMat
    
    def TOnMesh(self, dim):
        return tf.vectorized_map( self.xFEChebyBind, tf.range(0, dim, dtype=floatf) )

    def meshToChebyU(self, funcOnMesh):
        '''
            funcOnMesh must be [numNodes] -- TMat is [dimU, numNoes]
            return coeffs [dimU, 1]
        '''
        print(self.TXFEMatU, funcOnMesh, self.wMesh)
        a = 2. /_PI * tf.math.reduce_sum(self.TXFEMatU * funcOnMesh * self.wMesh , axis=1, keepdims=True) * (self.xFE[-1] - self.xFE[0])/self.xFE.shape[0]
        OneOver2Elm0 = tf.concat([[0.5], tf.constant(1., shape=[self.dimU-1])], 0)[:,None]
        return a * OneOver2Elm0
    
    @staticmethod
    def getChebyCoeffs_fromPoints( x, y, deg):
        return np.polynomial.chebyshev.chebfit(x, y, deg, rcond=None, full=False, w=None)
    
    @staticmethod
    def trigCheb(x : tf.Tensor , n : tf.Tensor) -> tf.Tensor :
        return tf.math.cos( n * tf.math.acos(x) )
        #return tf.math.real(tf.exp(tf.complex(0., 2.*_PI / 2. * n * x)))

    def xFEChebyBind(self, n):
        return self.trigCheb(self.xFE, n)
    

    '''Gaussian Propagation into Cheby Funcs'''
    def unscentedGaussTransf(self, mean, var, transform = tfp.bijectors.Softplus(), k = 1 ):
        d   = 1.
        sqScale = tf.sqrt( (1 + d) * var )
        X0 = mean
        X1 = X0 + sqScale
        X2 = X0 - sqScale
        W0 = k/(d+k)
        W12 = 1/(2*(d+k))
        Y0 = transform(X0)
        Y1 = transform(X1)
        Y2 = transform(X2)
        Ybar = W0 * Y0 + W12 * Y1 + W12 * Y2
        Yvar = W0 * (Y0-Ybar)**2 + W12 * (Y1-Ybar)**2 + W12 * (Y2-Ybar)**2 
        return Ybar, Yvar
        
    def gaussChebyGaussCoeffs(self, mean, var, x=None):
        '''mean, var are Tensor [numCoeffs]'''
        if x is None:
            Tx = tf.vectorized_map( self.xFEChebyBind, tf.range(0, mean.shape[0], dtype=floatf) )
        else:
            Tx        = self.genChebyT(mean.shape[0], x)
        meanCheby = tf.matmul(Tx, mean[:,None], transpose_a=True)
        diagCovCheby = tf.einsum("in,ij,jn->n", Tx, tf.linalg.diag(var), Tx)
        return tf.reshape(meanCheby, [-1]), diagCovCheby
       


if __name__ == '__main__':
    pass
