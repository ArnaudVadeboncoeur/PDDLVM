
from operator import gt
from turtle import shape

from numpy import gradient
from .utilities import *
from scipy import optimize

class Solver():

    def __init__(self, pdeState, xFE, isNonLinear = False):
        self.xFE       = xFE
        self.numFElems = self.xFE.shape[0] - 1
        self.numNodes  = self.numFElems+1

        self.pdeState  = pdeState
        self.u_trial   = tf.constant(0., shape=self.xFE.shape, dtype=floatf)

        self.elmCent = tffloat(self.xFE[:-1] + np.diff(self.xFE) / 2.)
        self.xEFh    = tffloat( np.diff(self.xFE) )
        print('xEFh = \n', self.xEFh)

        self.linElmMat      = tf.constant( [ [1., -1.], [-1., 1.] ], dtype=floatf )
        self.linElmMatRavel = tf.constant( [ 1., -1., -1., 1. ], dtype=floatf )
        self.zeroColumn     =  tf.constant(0.,  shape=[self.numFElems+1], dtype=floatf )

        
        self.isNonLinear =  isNonLinear 
        if self.isNonLinear == False:
            self.stress_strain = self.stress_strain_linear
            self.solve_tf      = self.solve_tf_linear
            self.assemble_tf   = self.assemble_tf
        else:
            self.stress_strain = self.stress_strain_nonLinear
            #! Options
                #self.stress_strain = self.stress_strain_nonLinear_depreacted
                #self.stress_strain = self.stress_strain_linear
                #self.stress_strain = self.stress_strain_nonLinear_ofU
            
            self.solve_tf_nonLinear= self.solve_tf_nonLinear_lsq
            #! Option
                #self.solve_tf_nonLinear = self.solve_tf_nonLinear_bfgs
            self.solve_tf      = self.solve_tf_nonLinear
            self.assemble_tf   = self.assemble_tf

        self.solve_tf_nonLinear = self.solve_tf_nonLinear_lsq
        self.assemble_tf = self.assemble_tf
    #!strain = du/dx; kappa(strain) = func
    def stress_strain_nonLinear_depreacted(self, index, kappa, u_trial):
        if index == self.u_trial.shape[0] - 1:
            dudx = (self.u_trial[index] - self.u_trial[index - 1]) / (self.xFE[index] - self.xFE[index-1])
        else:
            dudx = ( self.u_trial[index + 1] - self.u_trial[index]) / (self.xFE[index - 1 ] - self.xFE[index])
        return dudx * kappa + 2.
    
    def stress_strain_nonLinear(self, index, kappa, u_trial):
        #print('using NONlinear stress-strain')
        dudx = ( u_trial[index] - u_trial[index+1]) / (self.xFE[index] - self.xFE[index+1])
        return (tf.keras.activations.sigmoid(tf.math.abs(dudx)) + kappa*5. ) /10.

    def stress_strain_nonLinear_ofU(self, index, kappa, u_trial):
        return (u_trial[index]+u_trial[index+1])/2. * kappa + 2. 
    
    def stress_strain_linear(self, index, kappa, u_trial):
        return kappa  
            
    
    def assemble_tf_internal(self, u : tf.Tensor):

        kappaVect    = self.pdeState.kappa_funCXFE()
        forcingVect  = self.pdeState.forcing_funXFE()
        aMat         = tf.constant(0., shape=[ self.numFElems+1, self.numFElems+1 ], dtype=floatf )
        fVec         = tf.constant(0., shape=[self.numFElems+1], dtype=floatf ) 

        for e in range( self.numFElems ):
            
            hFE    = self.xEFh[e]
            kappaE = kappaVect[e]
            stiffness_of_kappa = self.stress_strain( e, kappaE, u )
            aMat   = tf.tensor_scatter_nd_add(aMat, [[e,e],[e,e+1],[e+1,e],[e+1,e+1]], stiffness_of_kappa * 1. / hFE * self.linElmMatRavel )

        # save the column corresponding to the right BC
        aMat_rightBC_col = tf.identity( aMat[:, self.numFElems]  )

        # assemble forcing vector
        for e in range( self.numFElems ):
            fL = forcingVect[ e ] 
            fR = forcingVect[ e + 1 ] 

            fVec = tf.tensor_scatter_nd_add(fVec, [[e]],      [ self.xEFh[e]   * ( 1./3. * fL + 1./6. * fR ) ] )
            fVec = tf.tensor_scatter_nd_add(fVec, [[e + 1]],  [ self.xEFh[e] * ( 1./6. * fL + 1./3. * fR ) ] )

            # enforce boundary conditions
            uL, uR = self.pdeState.dirichlet_fun( )    

            fVec -= uL * aMat[ : , 0 ] 
            fVec -= uR * aMat[ : , self.numFElems ] 

            fVec = tf.tensor_scatter_nd_update(fVec, [[0]], [uL] )
            fVec = tf.tensor_scatter_nd_update(fVec, [[self.numFElems]], [uR] )

            aMat = tf.tensor_scatter_nd_update(aMat, [[0]], [self.zeroColumn])
            aMat = tf.tensor_scatter_nd_update(aMat, [[self.numFElems]], [self.zeroColumn])
            aMat = tf.transpose( tf.tensor_scatter_nd_update(tf.transpose(aMat), [[0]], [self.zeroColumn])  )
            aMat = tf.transpose( tf.tensor_scatter_nd_update(tf.transpose(aMat), [[self.numFElems]], [self.zeroColumn])  )
            aMat = tf.tensor_scatter_nd_update(aMat, [[0,0]], [tffloat(1.)])
            aMat = tf.tensor_scatter_nd_update(aMat, [[self.numFElems, self.numFElems]], [tffloat(1.)])

        return aMat, fVec

    def assemble_tf(self, u):
        self.u_trial = u
        aMat, fVec = self.assemble_tf_internal( u )
        return aMat, fVec

    def assemble_tf_noGrad(self): return tf.stop_gradient( self.assemble_tf( ) )


    #=================== FE Discrete Linear Solver ===========================


    def solve_tf_linear(self, u=None):
        if self.isNonLinear == True:
            print('WARNING: PDE is NON-LINEAR -- THIS IS A LINEAR SOLVE METHOD')
        aMat, fVec = self.assemble_tf_internal( self.u_trial )
        u = tf.linalg.solve(aMat, fVec[:,None])
        return tf.squeeze(u)


    #=================== FE Discrete BFGS NONLinear Solver ===================

    def residualScalar(self, u):
        self.u_trial = u
        A, f = self.assemble_tf( u )
        residual = tf.linalg.norm( A @ self.u_trial[:,None] - f[:,None] )**2.
        return residual

    def valGrad(self, u):
        with tf.GradientTape() as tape:
            tape.watch( u )
            res = self.residualScalar(u)
        grad = tape.gradient(res, u )
        return res, grad
        
    def solve_tf_bfgs(self, u_0=None):

        if u_0 is None:
            u_0 = (tf.constant(0.,shape=self.u_trial.shape,dtype=floatf))
        valGrad = tf.function(self.valGrad)
        self.optim_results = tfp.optimizer.bfgs_minimize(
            valGrad, u_0, tolerance=1e-08, x_tolerance=0,
            f_relative_tolerance=0, initial_inverse_hessian_estimate=None,
            max_iterations=500, parallel_iterations=1, stopping_condition=None,
            validate_args=True, max_line_search_iterations=50, 
            name=None
        )
        return self.optim_results.position
    
    #=================== FE Discrete lsq NONLinear Solver ====================

    def residualVect_lsq(self, u):
        #u = tffloat(u)
        A, f = self.assemble_tf( u )
        #residual = self.preconditioner @ ( A @ u[:,None] - f[:,None] )
        residual = A @ u[:,None] - f[:,None] 
        return tf.squeeze(residual)
    
    def Jac(self, u):
        #u = tffloat(u)
        with tf.GradientTape() as tape:
            tape.watch( u )
            residual = self.residualVect_lsq(u)
        jac = tape.jacobian(residual, u )
        return jac

    
    def solve_tf_nonLinear_lsq(self, u_0=None, preconditioner = None):           
        if u_0 is None:
            u_0 = (tf.constant(0.,shape=self.u_trial.shape,dtype=floatf))
        if preconditioner is not None:
            self.preconditioner = preconditioner
        else:
            self.preconditioner = tf.eye(u_0.shape[0], dtype=floatf)

        JacTF = tf.function(self.Jac)
        ResTF = tf.function(self.residualVect_lsq)
        def JacWrapper(x): return JacTF(tffloat(x))
        def residualVect_lsqWrapper(x): return ResTF(tffloat(x))
        
        self.optim_results = optimize.least_squares(residualVect_lsqWrapper, u_0, jac=JacWrapper, 
                                     method='lm',
                                     loss='linear', xtol=1e-08)
        return tffloat(self.optim_results.x)

    #=================== FE Spectral BFGS NONLinear Solver ====================

    def residualScalar_FEspectral(self, uc):
        uFE = tf.squeeze(self.pdeState.u_funXFE( uc[:,None] ))
        A, f = self.assemble_tf( uFE )
        residual = tf.linalg.norm( A @ uFE[:,None] - f[:,None] ) ** 2
        return residual

    def valGrad_spec(self, u):
        with tf.GradientTape() as tape:
            tape.watch( u )
            res = self.residualScalar_FEspectral( u )
        grad = tape.gradient(res, u )
        return res,grad
        
    def solve_tf_FESpectral_bfgs(self, u_c=None):
        if u_c is None:
            u_c = tf.constant(0., shape=[self.pdeState.dimU])        
        valGrad_spec = tf.function(self.valGrad_spec)
        self.optim_results = tfp.optimizer.bfgs_minimize(
            valGrad_spec, u_c, tolerance=1e-08, x_tolerance=1e-8,
            f_relative_tolerance=1e-8, initial_inverse_hessian_estimate=None,
            max_iterations=500, parallel_iterations=1, stopping_condition=None,
            validate_args=True, max_line_search_iterations=50, 
            name=None
        )
        return self.optim_results.position

    #=================== FE spectral lsq NONLinear Solver lsq =====================

    def residualVect_FEspectral(self, uc):
        uc_trial = tffloat(uc)
        uFE = tf.squeeze(self.pdeState.u_funXFE( uc_trial[:,None] ))
        A, f = self.assemble_tf( uFE )
        residual = A @ uFE[:,None] - f[:,None]
        print(residual)
        return tf.squeeze(residual)

    def residualVect_FEspectral_coeffSpace(self, uc):
        uc_trial = tffloat(uc)
        with tf.GradientTape() as tape:
            tape.watch(self.elmCent)
            tape.watch(uc_trial)
            u_of_x = self.pdeState.genChebyFun( uc_trial[:,None], self.elmCent )

        self.gradUdx = tape.gradient(u_of_x, self.elmCent )
        uFE = tf.squeeze(self.pdeState.u_funXFE( uc_trial[:,None] ))
        A, f = self.assemble_tf( uFE )

    def Jac_spectral(self, uc):
        uc_trial = tffloat(uc)
        with tf.GradientTape() as tape:
            tape.watch(uc_trial)
            residual = self.residualVect_FEspectral(uc_trial)
        jac = tape.jacobian(residual, uc_trial)
        #print('jac =', jac )
        return jac
    
    def solve_tf_nonLinear_lsq_spectral(self, u_c=None):
        
        if u_c is None:
            u_c = tf.constant(1.,shape=[self.pdeState.dimU], dtype=floatf)

        self.optim_results = optimize.least_squares(self.residualVect_FEspectral, u_c , 
                                    jac=self.Jac_spectral, 
                                    method='lm',
                                    loss='linear', xtol=1e-15, ftol=1e-15, gtol=1e-15)
        return self.optim_results.x
        

        
if __name__ == '__main__':

    #================= Test for Solver =========================
    from pddlvm.poisson1DPDEState import PDE_State
    omegaLength = 2.0
    numFElems   = 20
    xFE = tf.constant( tf.linspace( 0., omegaLength, numFElems + 1) ) - 1.

    pdeState = PDE_State(xFE, dimK=1, dimU=3, dimF=1)

    #LINEAR TEST
    print("TEST LINEAR")
    solver = Solver( pdeState, xFE, isNonLinear=False )
    print(pdeState.dimK, pdeState.dimU)
    dudx = tf.cast(tf.linspace(0., 10., 200), floatf)
    kappa = tf.constant(0., dtype=floatf)

    u_0  = tf.constant(solver.xFE **2 +1., shape=solver.u_trial.shape, dtype=floatf)
    plt.plot(solver.xFE, solver.solve_tf_linear(), c = 'b', label='inversion solve linear')
    u_sln = solver.solve_tf_nonLinear( )
    print(solver.optim_results)
    plt.plot(solver.xFE, u_sln,c='g', label='non-linear solve')
    plt.legend()
    plt.show()

    print("TEST NON-LINEAR")
    soft = tfb.Softplus()
    y = (tf.keras.activations.sigmoid(tf.math.abs(dudx*2.)/5.) + soft(kappa) ) / 20. 
    plt.plot(dudx, y,label='stress-strain')
    plt.legend()
    plt.show()
    solver = Solver( pdeState, xFE, isNonLinear=True )
    print(pdeState.dimK, pdeState.dimU)
    dudx = tf.cast(tf.linspace(0., 10., 200), floatf)
    kappa = tf.constant(0., dtype=floatf)

    plt.plot(solver.xFE, solver.solve_tf_linear(), c = 'b', label='inversion solve linear')
    u_sln = solver.solve_tf_nonLinear( )
    print(solver.optim_results)
    plt.plot(solver.xFE, u_sln,c='g', label='non-linear solve')
    plt.legend()
    plt.show()

    omegaLength = 2.0
    numFElems   = 20
    xFE = tf.constant( tf.linspace( 0., omegaLength, numFElems + 1) ) - 1.

    pdeState = PDE_State(xFE, dimK=1, dimU=5, dimF=1)

    #LINEAR TEST
    print("TEST LINEAR")
    solver = Solver( pdeState, xFE, isNonLinear=True )

    plt.plot(solver.xFE, solver.solve_tf_linear(), c = 'b', label='inversion solve linear')

    u_0  = tf.constant(1., shape=[pdeState.dimU], dtype=floatf)
    uc_sln = solver.solve_tf_nonLinear_lsq_spectral( u_c = u_0)
    print(solver.optim_results)
    plt.plot(solver.xFE, pdeState.u_funXFE( uc_sln[:,None] ),c='g', label='non-linear solve')
    plt.legend()
    plt.show()