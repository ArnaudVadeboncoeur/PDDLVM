
#from msvcrt import kbhit
from .utilities import *
from .NeuralNet import NeuralN

_0        = tf.constant(0., dtype=floatf)

class VE():
    '''Variational Emulator'''
    def __init__(self, pdeState, solver):
        self.pdeState  = pdeState
        self.solver   = solver
        self.dimU     = self.pdeState.dimU
        self.dimK     = self.pdeState.dimK
        self.dimF     = self.pdeState.dimF
        self.dimBl    = 1
        self.dimBr    = 1
        self.epsilon_r = tf.Variable(tf.constant(1e-2, dtype=floatf), trainable=False)
        self.epsilon_y = tf.Variable(tf.constant(1e-1, dtype=floatf), trainable=False)

        self.LatentKappaPrior   = None
        self.LatentForcingPrior = None
        self.LatentBlPrior      = None
        self.LatentBrPrior      = None

        
        self.dimY      = self.dimU
        self.dataY     = tffloat([0.])
        self.dataForcingPrior = None
        self.dataKappaPrior   = None
        self.dataBlPrior      = None
        self.dataBrPrior      = None

        self.logProbPhysResidual = self.physicsPenalty_Cheby
        
        self.log2pi    = tf.math.log(2. * tf.constant(np.pi, dtype=floatf))
        self.n_sgd     = 1
        self.miniBatch = 1
               
        
    def setModePrior(self):
        self.mode = 'prior'
        self.setPriorOnlyTrainable()
    def setModePost(self):
        self.mode = 'post'
        self.setPostOnlyTrainable()
    def setModeCouple(self):
        self.mode = 'post'
        self.setPriorPostTrainable()
    
    def setPriorOnlyTrainable(self):
        self.trainableVariables = [self.priorModels['trainable_variables']]
    def setPostOnlyTrainable(self):
        self.trainableVariables = [self.postModels['trainable_variables']]
    def setPriorPostTrainable(self):
        self.trainableVariables = [self.priorModels['trainable_variables'], self.postModels['trainable_variables']]

        
    def initPriorNetworks(self, U_KFBlBr = None, Z_UF = None):

        if U_KFBlBr is None:
            dimKFBlBr = self.dimK + self.dimF + self.dimBl + self.dimBr
            self.U_KFBlBr  = NeuralN('U_of_KFBlBr', 
                            np.array([dimKFBlBr,
                                        20, 20, 20, self.dimU])
                            )
        else:
            self.U_KFBlBr = U_KFBlBr

        if Z_UF is None:
            k  = self.dimK
            bl = k  + self.dimBl
            br = bl + self.dimBr
            self.Z_UF     = NeuralN('Z_of_UF',
                                    np.array([self.dimU + self.dimF, 20, 20, 20,
                                            self.dimK + self.dimBl + self.dimBr]),
                                    multOutputLayout = {'K': tf.range(0,k),
                                                        'Bl':tf.range(k,bl),'Br':tf.range(bl,br)}
                                )
        else:
            self.Z_UF = Z_UF
        
        self.U_KFBlBr.compile()
        self.Z_UF.compile()

        self.varListPriorNets = [self.U_KFBlBr.trainable_variables,
                                 self.Z_UF.trainable_variables
                                ]
        
        self.priorModels = {self.U_KFBlBr.NN_name:self.U_KFBlBr, 
                            self.Z_UF.NN_name:self.Z_UF, 
                            'trainable_variables':self.varListPriorNets}
        
    def initPostNetworks(self, Y_U=None, KBlBr_Y=None):
        
        if Y_U is None:
            self.Y_U  = NeuralN('Y_of_U', np.array([self.dimU, 101,101, self.dimY]),
                            transf = tfb.JointMap([tfb.Identity()])
                            )   
        else:
            self.Y_U = Y_U


        if KBlBr_Y is None:
            k  = self.dimK
            bl = k  + self.dimBl
            br = bl + self.dimBr
                
            self.KBlBr_Y = NeuralN('KBlBr_of_Y', 
                            np.array([self.dimY,
                                    50,50,50,50,50,50,50,50, self.dimK + self.dimBl + self.dimBr]),
                            transf = tfb.JointMap([tfb.Identity()]), 
                            multOutputLayout = {'K': tf.range(0,k),
                                                'Bl':tf.range(k,bl),'Br':tf.range(bl,br)}
                        )
        else:
            self.KBlBr_Y = KBlBr_Y
        
        self.Y_U.compile()
        self.KBlBr_Y.compile()      
        self.varListPostNets = [
                                self.Y_U.trainable_variables,
                                self.KBlBr_Y.trainable_variables
                               ]
        
        self.postModels = { 
                            self.Y_U.NN_name:self.Y_U, 
                            self.KBlBr_Y.NN_name:self.KBlBr_Y,
                            'trainable_variables':self.varListPostNets}
        
       
    @staticmethod
    def reparameterize( mean, logvar):
        eps = tf.random.normal(shape=[mean.shape[0], mean.shape[1]], dtype=floatf)
        z = eps * tf.exp(logvar * .5) + mean
        tf.debugging.assert_all_finite(z, 'z reparam not finite')
        return z    

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + self.log2pi), axis=raxis)
    
    def physicsPenalty(self, u, epsilon):
        A, f = self.solver.assemble_tf( tf.reshape(u, [-1]) )
        d = A @ u[0,:][:,None] -  f[:,None]

        penalty = - 0.5 * ( A.shape[0] * tf.math.log(epsilon**2) 
                            + tf.squeeze( tf.linalg.norm(d, axis = 0)**2/epsilon**2 ) + self.log2pi )
        return penalty
    
    def physicsPenalty_Cheby(self, uc, epsilon):
        u    = self.pdeState.u_funXFE( tf.reshape(uc,[-1, 1]))
        A, f = self.solver.assemble_tf( u )
        d    = A @ tf.reshape(u,[-1,1]) -  f[:,None]
        penalty = - 0.5 * ( A.shape[0] * tf.math.log(epsilon**2) 
                            + tf.squeeze( tf.linalg.norm(d, axis = 0)**2/epsilon**2 ) + self.log2pi )
        return penalty
    
    def compute_nelbo_prior(self, iteration):

        nelboSGD     = 0.

        i = iteration % self.dataY.shape[0] 
        for j in range(self.n_sgd):

            f_i       = self.LatentForcingPrior.sample(sample_shape=(1))
            k_i       = self.LatentKappaPrior.sample(sample_shape=(1))
            bdLeft_i  = self.LatentBlPrior.sample(sample_shape=(1))
            bdRight_i = self.LatentBrPrior.sample(sample_shape=(1))

            self.pdeState.forcing = tf.reshape(f_i, [-1]) 
            self.pdeState.kappa   = tf.reshape(k_i, [-1]) 
            self.pdeState.bdLeft  = tf.reshape(bdLeft_i, []) 
            self.pdeState.bdRight = tf.reshape(bdRight_i, []) 

            mu_alpha, logvar_alpha = self.priorModels['U_of_KFBlBr'].Map( [k_i, f_i, bdLeft_i, bdRight_i] ) 

            us  = self.reparameterize(mu_alpha, logvar_alpha)

            sampleZ         = tf.concat([k_i, bdLeft_i, bdRight_i],1)
            mean_beta_Z,   logvar_beta_Z = self.priorModels['Z_of_UF'].Map( [us, f_i] )
            logP_beta_Z_uf  = self.log_normal_pdf(sampleZ, mean_beta_Z, logvar_beta_Z)

            logP_alpha_u_Z  = self.log_normal_pdf( us , mu_alpha, logvar_alpha)

            logPr_Z         = self.logProbPhysResidual( us, self.epsilon_r )

            elbo            = tf.reduce_mean( logPr_Z  + logP_beta_Z_uf  - logP_alpha_u_Z  )

            nelboSGD       += - elbo


        nelboSGD  *= 1./self.n_sgd 

        return nelboSGD 
    
    def compute_nelbo_post(self, iteration):

        nelboSGD     = 0.

        i = iteration % self.dataY.shape[0] 
        for j in range(self.n_sgd):

            print('at Post Mode')
            mean_phi, logvar_phi = self.postModels['KBlBr_of_Y'].Map( [self.dataY[i,:][None,:]] )
            Zs                   = self.reparameterize(mean_phi, logvar_phi)
            logq_phi_Z_y         = self.log_normal_pdf( Zs, mean_phi, logvar_phi )

            k_i       = tf.gather(Zs, self.postModels['KBlBr_of_Y'].multOutputLayout['K'], axis = 1 )
            bdLeft_i  = tf.gather(Zs, self.postModels['KBlBr_of_Y'].multOutputLayout['Bl'], axis = 1 )
            bdRight_i = tf.gather(Zs, self.postModels['KBlBr_of_Y'].multOutputLayout['Br'], axis = 1 )
            
            f_i  =  tf.random.normal( mean=self.dataForcingPrior[0][i,:], stddev = tf.exp(self.dataForcingPrior[1][i,:]*0.5), shape=[1,self.dimF], dtype=floatf  )                  
            
            mean_alpha, logvar_alpha = self.priorModels['U_of_KFBlBr'].Map( [k_i, f_i, bdLeft_i, bdRight_i] )
            us                       = self.reparameterize( mean_alpha, logvar_alpha )
            logqU_alpha_ZF           = self.log_normal_pdf( us, mean_alpha, logvar_alpha )

            self.pdeState.forcing = tf.reshape(f_i, [-1]) 
            self.pdeState.kappa   = tf.reshape(k_i, [-1]) 
            self.pdeState.bdLeft  = tf.squeeze(bdLeft_i)
            self.pdeState.bdRight = tf.squeeze(bdRight_i) 

            uFE         = tf.reshape(self.pdeState.u_funXFE(tf.reshape(us, [-1,1])) , [1,-1] )
            logPr_y_u   = self.log_normal_pdf(self.dataY[i,:][None,:], uFE, 2*tf.math.log(self.epsilon_y))

            mean_theta, logvar_theta = self.postModels['Y_of_U'].Map( [us] )

            logP_theta_y_u = self.log_normal_pdf(self.dataY[i,:][None,:], mean_theta, logvar_theta )
          
            logPKappa = self.log_normal_pdf( k_i, self.dataKappaPrior[0][i,:], self.dataKappaPrior[1][i,:] )
            logPBl    = self.log_normal_pdf( bdLeft_i, self.dataBlPrior[0][i,:], self.dataBlPrior[1][i,:] )
            logPBr    = self.log_normal_pdf( bdRight_i, self.dataBrPrior[0][i,:], self.dataBrPrior[1][i,:] )
            logPZ    = tf.reduce_sum(logPKappa + logPBl + logPBr)
            
            elbo      = tf.reduce_mean( logPr_y_u + logP_theta_y_u + logPZ
                                       - logq_phi_Z_y - logqU_alpha_ZF  )

            nelboSGD     -= elbo

        nelboSGD       *= 1./self.n_sgd

        return nelboSGD   


if __name__ == '__main__':
    
    from poisson1DPDEState import PDE_State
    from poisson1DSolver import Solver

    pdeState = PDE_State(dimK=1, dimU=3, dimF=1)

    omegaLength = 2.0
    numFElems   = 20
    xFE = tf.constant( np.linspace( 0., omegaLength, numFElems + 1) ) - 1.
    solver = Solver( pdeState, xFE, isNonLinear=False )

    model = VE( pdeState, solver )
    model.initPriorNetworks()
    model.initPostNetworks() 
    model.setModePrior()