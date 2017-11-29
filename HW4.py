import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as const

import pickle, corner

#####################################################################################################################

##### General MCMC Fitting Functions -- Likelihood, Posterior P, Proposal #####

## Log Likelihood function
def logLike( p, xdata, ydata, yerr ):

    ymodel = model( p, xdata ) # Get model values

    return -0.5 * np.sum( ( ( ydata - ymodel ) / yerr ) ** 2.0 + np.log( 2 * np.pi * yerr ** 2.0 ) )

## Log Posterior
def logProb( p, xdata, ydata, yerr ):

    # Just return sum of log likelihood and log priors (defined below)
    return logLike( p, xdata, ydata, yerr ) + logPrior( p )

## Proposal
def proposalq( x, var ):

    # Gaussian centered on previous sample with variance as input
    return np.random.normal( x, np.sqrt(var), 1 )[0]

##### Model specific functions - model and priors #####

## RV Curve Model
def model( p, x ):

    # Assumes P in days, Mp in Mjup, and t0 in days (JD)
    
    P, Mp, t0 = p
    Ms        = 1.13 * u.solMass.to('kg') # Looked up stellar mass

    Psec = P * u.d.to('s')
    Mpkg = Mp * u.Mjup.to('kg')

    # RV amplitude
    A = Mpkg / Ms * ( 2 * np.pi * const.G.value * Ms / Psec ) ** ( 1./3. )

    return A * np.sin( 2 * np.pi * ( x - t0 ) / P )

## Prior
def logPrior( p ):

    P, Mp, t0 = p

    if P <= 0 or Mp <= 0: # Don't let P or Mp get below zero
        return -np.inf
    else:
        return 0.0

##### Fitting #####

data  = pd.read_table( 'HD209458.lst', delim_whitespace = True )
xarr  = data.Date
yarr  = data.RV
eyarr = data.eRV

p0   = np.array( [ 3.525, 0.71, 2451411.4 ] ) # Initial guesses for P, Mpsini, t0
pvar = np.array( [ 1e-8, 1e-5, 8e-4 ] )       # Proposal variances

## Set up for the fit
N          = 205000                  # Number of samples (5000 will be thrown out as burnin, which is overkill but ok)
dims       = 3                       # Number of variables
samples    = np.zeros( ( N, dims ) ) # Set up array for samples
samples[0] = p0                      # Set first sample to initial guess
acc_rej    = np.zeros( N - 1 )       # Set up array to calculate acceptance percentage

## Perform sampling
for i in range( 1, N ):

    # Pick random variable to sample in
    dim   = np.random.randint( 0, dims )

    xprev      = samples[i-1] # Previous sample
    xnext      = xprev.copy()
    xnext[dim] = proposalq( xprev[dim], pvar[dim] ) # Next sample

    r  = np.random.random() # Pick random variable that sets how much larger than P(prev) P(next) must be
    a1 = logProb( xnext, xarr, yarr, eyarr ) - logProb( xprev, xarr, yarr, eyarr )

    if a1 > np.log(r): # Accept the new sample
        samples[i] = xnext
        acc_rej[i-1] = 1.0
    else: # Reject new sample
        samples[i] = xprev

## Slight clean up
wburnin = samples.copy() # Full samples array
samples = samples[5000:] # Get rid of burn in

# Output the three arrays (acceptance, samples, all samples)
pickle.dump( acc_rej, open( 'acc_rej.pkl', 'wb' ) )
pickle.dump( samples, open( 'samples.pkl', 'wb' ) )
pickle.dump( wburnin, open( 'wburnin.pkl', 'wb' ) )

##### Results and Plots #####

# Acceptance percent
print acc_rej.sum() / acc_rej.size, '% of steps were accepted.\n'

# Calculate the median, 16th percentile, and 84th percentile
meds = np.median( samples, axis = 0 )
p16  = np.percentile( samples, 16.0, axis = 0 )
p84  = np.percentile( samples, 84.0, axis = 0 )

## Corner plot
def lvl( sigma ): # Calculates level percentile for a 2d gaussian sigma
    return 1 - np.exp( - sigma ** 2.0 / 2 )

fig = corner.corner( samples, quantiles = [ 0.16, 0.50, 0.84 ], levels = [ lvl(1.0), lvl(2.0), lvl(3.0) ] )
plt.savefig('corner.pdf')

## Phased RV Curve

# Data phase array
phase  = ( ( xarr - meds[2] ) / meds[0] * 2 * np.pi ) % ( 2 * np.pi ) / 2 / np.pi

# Model in terms of phase
oneP   = np.linspace( meds[2], meds[2] + meds[0], 1000 )
onePph = ( oneP - meds[2] ) / meds[0]
onePrv = model( meds, oneP )

# Plot
plt.clf()
plt.plot( onePph, onePrv, 'r-' )
plt.errorbar( phase, yarr, yerr = eyarr, fmt = 'ko', capsize = 3.0 )
plt.xlabel( 'Orbital Phase')
plt.ylabel( 'Stellar RV (m/s)' )
plt.savefig('phaserv.pdf')
