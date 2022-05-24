import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tfb = tfp.bijectors
tfm = tf.math
floatf = tf.float32

def updateTfFloat(newFloat): 
    global floatf
    floatf = newFloat
    
def tffloat(x): return tf.constant(x, dtype=floatf)
def tfint(x): return tf.constant(x, dtype=tf.int32)

#! Some constants
_0 = tffloat(0.)
_PI = tffloat(np.pi)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12
          })
# latex font definition
plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})
from matplotlib.transforms import Bbox

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)