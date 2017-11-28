import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as const

import pdb, pickle, corner

# def proposal( x ):

#     var = 5.0

#     return np.random.normal( x, np.sqrt( var ), 1 )[0]

# def pxy( xy ):

#     # xy   = np.array( [ x, y ] )
#     mu   = np.array( [ 0.0, 0.0 ] )
#     cov  = np.array( [ [ 2.0, 1.2 ], [ 1.2, 2.0 ] ] )
#     icov = np.linalg.inv( cov )

#     A   = 1 / ( 2 * np.pi * np.sqrt( np.linalg.det( cov ) ) )
#     e   = -0.5 * np.dot( np.dot( xy - mu, icov ), xy - mu )

#     return A * np.exp( e )

# N       = 50000
# samples = np.zeros( ( 2, N ) )
# samples[:,0] = np.array([8.0,10.0])
# dims    = 2
# acc_rej = np.zeros( N - 1 )

# for i in range( 1, N ):

#     dim   = np.random.randint( 0, dims )

#     xprev      = samples[:,i-1]
#     xnext      = xprev.copy()
#     xnext[dim] = proposal( xprev[dim] )

#     r  = np.random.random()
#     a1 = pxy( xnext ) / pxy( xprev )

#     if a1 > r:
#         samples[:,i] = xnext
#         acc_rej[i-1]   = 1.0
#     else:
#         samples[:,i] = xprev

# print acc_rej.sum() / acc_rej.size

# plt.clf()
# plt.plot( samples[0], samples[1], 'k,' )
# plt.show()

# plt.clf()
# plt.hist( samples[0], bins = 100, histtype = 'step', color = 'k', normed = True )
# plt.show()

# plt.clf()
# plt.hist( samples[1], bins = 100, histtype = 'step', color = 'k', normed = True )
# plt.show()

#####

### MCMC Fitting functions

def logLike( p, xdata, ydata, yerr ):

    ymodel = model( p, xdata )

    return -0.5 * np.sum( ( ( ydata - ymodel ) / yerr ) ** 2.0 + np.log( 2 * np.pi * yerr ** 2.0 ) )

def logProb( p, xdata, ydata, yerr ):

    return logLike( p, xdata, ydata, yerr ) + logPrior( p )

def proposalq( x, var ):

    return np.random.normal( x, np.sqrt(var), 1 )[0]

### Model specific functions

# Straight line

# def model( p, x ):

#     a, b = p

#     return a * x + b

# def logPrior( p ):

#     a, b = p

#     if -20 <= a <= 25 and -5 <= b <= 5:
#         return 0.0
#     else:
#         return -np.inf

# preal = np.array( [ 4.32, 0.543 ] )
# xarr  = np.random.uniform( -5, 14, 25 )
# yarr  = model( preal, xarr ) + np.random.normal( 0.0, 2.0, xarr.size )

# p0    = np.array( [ 0.0, 0.0 ] )

# 1D Gaussian

# def model( p, x ):

#     m, s, A, b = p

#     return A * np.exp( - ( x - m ) ** 2.0 / ( 2.0 * s ** 2.0 ) ) + b

# def logPrior( p ):

#     m, s, A, b = p
    
#     if s <= 0:
#         return -np.inf
#     else:
#         return 0.0

# preal = np.array( [ 3.0, 1.35, 18.78, 4.0 ] )
# xarr  = np.random.uniform( -8.0, 10.0, 30 )
# yarr  = model( preal, xarr ) + np.random.normal( 0.0, 2.0, xarr.size )

# p0    = np.array( [ 0.0, 0.0, 0.0, 0.0 ] )

# RV Curve Model

def model( p, x ):

    # Assumes P in days, Mp in Mjup, and t0 in days (JD)
    
    P, Mp, t0 = p
    Ms        = 1.13 * u.solMass.to('kg')

    Psec = P * u.d.to('s')
    Mpkg = Mp * u.Mjup.to('kg')
    
    A = Mpkg / Ms * ( 2 * np.pi * const.G.value * Ms / Psec ) ** ( 1./3. )

    return A * np.sin( 2 * np.pi * ( x - t0 ) / P )

def logPrior( p ):

    P, Mp, t0 = p

    if P <= 0 or Mp <= 0:
        return -np.inf
    else:
        return 0.0

data = pd.read_table( 'HD209458.lst', delim_whitespace = True )
xarr = data.Date
yarr = data.RV
eyarr = data.eRV

p0   = np.array( [ 3.525, 0.71, 2451411.4 ] )
# p0   = np.array( [ 5.0, 1.0, 2451411.4 ] )
pvar = np.array( [ 1e-8, 1e-5, 1e-5 ] )

N          = 505000
dims       = 3
samples    = np.zeros( ( N, dims ) )
samples[0] = p0
acc_rej    = np.zeros( N - 1 )

for i in range( 1, N ):

    dim   = np.random.randint( 0, dims )

    xprev      = samples[i-1]
    xnext      = xprev.copy()
    xnext[dim] = proposalq( xprev[dim], pvar[dim] )

    r  = np.random.random()
    a1 = logProb( xnext, xarr, yarr, eyarr ) - logProb( xprev, xarr, yarr, eyarr )

    if a1 > np.log(r):
        samples[i] = xnext
        acc_rej[i-1] = 1.0
    else:
        samples[i] = xprev

burnin  = samples[:5000].copy()
samples = samples[5000:]

pickle.dump( acc_rej, open( 'acc_rej.pkl', 'wb' ) )
pickle.dump( samples, open( 'samples.pkl', 'wb' ) )
pickle.dump( burnin, open( 'burnin.pkl', 'wb' ) )

print acc_rej.sum() / acc_rej.size

meds = np.median( samples, axis = 0 )
p16  = np.percentile( samples, 16.0, axis = 0 )
p84  = np.percentile( samples, 84.0, axis = 0 )

print meds - p16
print meds
print p84 - meds

xplot = np.linspace( xarr.min(), xarr.max(), 1000000 )
yplot = model( meds, xplot )

plt.clf()
plt.plot( xarr, yarr, 'k.' )
plt.plot( xplot, yplot, 'r:' )
plt.show()

fig = corner.corner( samples )
plt.show()

# plt.clf()
# fig = plt.figure()
# for i in range( dims ):
#     j = i + 1
#     fig.add_subplot( dims, dims, dims * i + j )
#     plt.hist( samples[:,i], bins = 20, histtype = 'step', color = 'k' )
# plt.show()

# plt.clf()
# fig = plt.figure()
# # plot the 2D histogram
# fig.add_subplot( 223 )
# #plt.plot( samples[:,0], samples[:,1], 'k,', alpha = 0.5 )
# plt.hist2d( samples[:,0], samples[:,1], bins = 40, cmap = plt.get_cmap('gray') )

# plt.axvline( x = p16[0], color = 'r', ls = ':' )
# plt.axvline( x = meds[0], color = 'r' )
# plt.axvline( x = p84[0], color = 'r', ls = ':' )

# plt.axhline( y = p16[1], color = 'b', ls = ':' )
# plt.axhline( y = meds[1], color = 'b' )
# plt.axhline( y = p84[1], color = 'b', ls = ':' )

# # plot the upper 1D histogram
# fig.add_subplot( 221 )
# plt.hist( samples[:,0], bins = 20, histtype = 'step', color = 'k' )

# plt.axvline( x = p16[0], color = 'r', ls = ':' )
# plt.axvline( x = meds[0], color = 'r' )
# plt.axvline( x = p84[0], color = 'r', ls = ':' )

# plt.xticks( [], [] ); plt.yticks( [], [] )
# # plot the right 1D histogram
# fig.add_subplot( 224 )
# plt.hist( samples[:,1], bins = 20, histtype = 'step', color = 'k' )

# plt.axvline( x = p16[1], color = 'b', ls = ':' )
# plt.axvline( x = meds[1], color = 'b' )
# plt.axvline( x = p84[1], color = 'b', ls = ':' )

# plt.yticks( [], [] ); plt.xticks( [], [] )

# fig.subplots_adjust( wspace = 0, hspace = 0 )

# plt.show()
