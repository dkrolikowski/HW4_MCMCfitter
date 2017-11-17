import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import pdb

def proposal( x ):

    var = 1e-1

    return np.random.normal( x, np.sqrt( var ), 1 )[0]

def pxy( xy ):

    # xy   = np.array( [ x, y ] )
    mu   = np.array( [ 0.0, 0.0 ] )
    cov  = np.array( [ [ 2.0, 1.2 ], [ 1.2, 2.0 ] ] )
    icov = np.linalg.inv( cov )

    A   = 1 / ( 2 * np.pi * np.sqrt( np.linalg.det( cov ) ) )
    e   = -0.5 * np.dot( np.dot( xy - mu, icov ), xy - mu )

    return A * np.exp( e )

N       = 50000
samples = np.zeros( ( 2, N ) )
dims    = 2

for i in range( 1, N ):

    dim   = np.random.randint( 0, dims )

    xprev      = samples[:,i-1]
    xnext      = xprev.copy()
    xnext[dim] = proposal( xprev[dim] )

    r  = np.random.random()
    a1 = pxy( xnext ) / pxy( xprev )

    if a1 > r:
        samples[:,i] = xnext
    else:
        samples[:,i] = xprev

plt.clf()
plt.plot( samples[0], samples[1], 'k,' )
plt.show()

plt.clf()
plt.hist( samples[0], bins = 100, histtype = 'step', color = 'k', normed = True )
plt.show()

plt.clf()
plt.hist( samples[1], bins = 100, histtype = 'step', color = 'k', normed = True )
plt.show()
