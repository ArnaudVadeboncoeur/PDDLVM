
from .utilities import *
from tensorflow import keras

class NeuralN(tf.keras.Model):
    
    def __init__(self, name, layout, transf = None, multOutputLayout = None ):
        super(NeuralN, self).__init__( )
        self.NN_name       = name
        self.layout        = layout #np.array([input, n_layer1, ..., n_layeri, output])
        self.multOutputLayout = multOutputLayout
        self.activation = 'swish'
        self.initializer = tf.keras.initializers.HeNormal()
        self.lower_alpha = tf.constant(1e-6, dtype=floatf)
        self.upper_w     = tf.constant(1., dtype=floatf)

        self.hiddenLayerType = keras.layers.Dense

        self._0 = tffloat(0.) 
        
        
        if transf is None:
            def identity(a): return a 
            self.transf_forward = identity
        else:
            self.transf_forward = transf
            
        self.initialize()
    
    
    def initialize(self):
        self.NN = keras.Sequential()

        for i in range(1, self.layout.shape[0] - 1):
          if i == 1: 
            print('first layer')
            self.NN.add(self.hiddenLayerType(self.layout[i], input_dim = self.layout[0], activation = self.activation, kernel_initializer=self.initializer))
          else:
            print('inner layer ', i)
            self.NN.add(self.hiddenLayerType(self.layout[i], activation = self.activation, kernel_initializer=self.initializer))
        print('last layer')
        self.NN.add(keras.layers.Dense(  self.layout[-1] * 2  ))
        self.NN.summary()

        
    def Map(self, varIn):
        
        #tf.debugging.assert_equal( len(varIn), self.layout.shape[0], message='Input wrong size for {}'.format(self.NN_name))
        
        varIn  = self.transf_forward(varIn)
        allVar = tf.constant(0., shape=[varIn[0].shape[0],1],dtype=floatf)
        for var in varIn:
            allVar = tf.concat([allVar, var], 1)
        allVar = allVar[:,1:]
        
        mapped = self.NN( allVar )
        
        mean, logvar = mapped[:,:self.layout[-1]], mapped[:,self.layout[-1]:]
        
        tf.debugging.assert_all_finite(mean, 'mean not finite')
        tf.debugging.assert_all_finite(logvar, 'logvar not finite')
        #logvar = self.upBoundw_log( logvar )
        logvar = self.dBoundwalpha_log( logvar )
        mean   = tf.cast(mean, floatf)
        logvar = tf.cast(logvar, floatf)
        return mean, logvar
    
    
    def upBoundw(self, X):
        X = tf.cast(X, floatf)
        p_plus  = tf.clip_by_value(X, self._0, X )
        p_minus = tf.clip_by_value(X, -tf.math.abs(X), self._0 )
        delta_plus  = tf.cast(X > self._0, floatf) 
        delta_minus = tf.cast(X <= self._0, floatf) 
        sig  = self.upper_w / 2. * tf.exp( p_minus ) * delta_minus +   self.upper_w / 2. * ( 2. - tf.exp(-p_plus) ) * delta_plus
        return sig

    
    def upBoundw_log(self, X):
        X = tf.cast(X, floatf)
        p_plus  = tf.clip_by_value(X, self._0, X )
        p_minus = tf.clip_by_value(X, -tf.math.abs(X), self._0 )
        delta_plus  = tf.cast(X > self._0, floatf) 
        delta_minus = tf.cast(X <= self._0, floatf) 
        logw_2 = tf.math.log( self.upper_w / 2.)
        sigH = (p_minus + logw_2) * delta_minus + ( tf.math.log(2. - tf.exp(-p_plus)) + logw_2 ) * delta_plus
        return sigH

    
    def dBoundwalpha(self, X):
        sig  = ( self.lower_alpha + (self.upper_w - self.lower_alpha) * tf.keras.activations.sigmoid(X) ) 
        return sig

    
    def dBoundwalpha_log(self, X):
        sigH = tf.math.log( self.dBoundwalpha(X) )
        return sigH

