"""

Required Parameters

Y  --> Observed flux array (2D: wavelength * time)
N_l  --> Number of wavelength bins
N_t  --> Number of time points
x_t  --> Time values (days, usually BJD_TDB or relative to mid-transit)
x_l  --> Wavelength values (Å or nm)

### Fixed Parameters of Transit Model
e  --> Eccentricity
w  --> Longitude of periastron (radians or degrees)

### Mean Function Parameters (to fit)
T0  --> Mid-transit time (days)
a  --> Scaled semi-major axis (a/R*)
b  --> Impact parameter
rho  --> Radius ratio Rp/R* (per wavelength)
q1  --> Kipping limb darkening parameter q1
q2  --> Kipping limb darkening parameter q2
Foot  --> Baseline flux normalization (per wavelength bin)
Tgrad  --> Linear baseline slope in time (per wavelength bin)

### Simulation Parameters (p_sim) (dictionary of true values for each parameter)
p_sim = {
T0  --> Central transit time
P  --> Orbital period (days)
a  --> Semi-major axis to stellar radius ratio (a/R*)
rho  --> Planet-to-star radius ratio (Rp/R*) per wavelength
b  --> Impact parameter
q1, q2  --> Kipping limb-darkening parameters (per wavelength, converted from u1, u2)
Foot  --> Baseline flux per wavelength
Tgrad  --> Baseline slope per wavelength
log_h  --> Kernel amplitude (log scale)
log_l_l  --> Length scale for wavelength axis (log scale)
log_l_t  --> Length scale for time axis (log scale)
log_sigma  --> White noise amplitude per wavelength (log scale)
}

### Mean Function Parameters 2D (mfp_2D) (dictionary of starting values for each parameter)
mfp_2D = {
T0  --> Central transit time
P  --> Orbital period (days)
a  --> Semi-major axis to stellar radius ratio (a/R*)
rho  --> Planet-to-star radius ratio (Rp/R*) per wavelength
b  --> Impact parameter
u1  --> Quadratic limb-darkening coefficient 1 (per wavelength)
u2  --> Quadratic limb-darkening coefficient 2 (per wavelength)
Foot  --> Baseline flux per wavelength
Tgrad  --> Baseline slope per wavelength
}

### Kernel Hyperparameters
log_h  --> Kernel amplitude (log scale)
log_l_l  --> Length scale for wavelength axis (log scale)
log_l_t  --> Length scale for time axis (log scale)
log_sigma  --> White noise term (log scale, per wavelength bin)

### Gaussian priors
a_mean  --> Mean of semi-major axis prior
a_std  --> Std dev of semi-major axis prior
u1_mean  --> Mean of limb-darkening coefficient u1 prior
u1_std  --> Std dev of limb-darkening coefficient u1 prior
u2_mean  --> Mean of limb-darkening coefficient u2 prior
u2_std  --> Std dev of limb-darkening coefficient u2 prior

### Bounds for mean hyperparameters (ideal)
min_log_l_l  --> Lower bound for log_l_l (set from wavelength bin spacing)
max_log_l_l  --> Upper bound for log_l_l (set from wavelength range)
min_log_l_t  --> Lower bound for log_l_t (set from time sampling)
max_log_l_t  --> Upper bound for log_l_t (set from total time span)
q1  --> [0, 1]
q2  --> [0, 1]
rho  --> [0, 1]
a  --> [0, 20]
T0  --> Flat prior (optionally restrict around Tc)
b  --> Flat prior (physically ~ [0, 1+Rp/R*])
Foot  --> Flat prior
Tgrad  --> Flat prior
log_h  --> [ln(1e-6), ln(1)]
log_l_l  --> [min_log_l_l, max_log_l_l]
log_l_t  --> [min_log_l_t, max_log_l_t]
log_sigma  --> [ln(1e-6), ln(1e-2)]

### Inference Settings
draws  --> Number of MCMC samples (default: 1000)
tune  --> Warm-up steps (default: 1000)
chains  --> Number of independent chains (default: 4)
cores  --> CPU cores (default: 1 for JAX compatibility)

"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os.path
import logging
import jaxoplanet
import arviz as az
import astropy.units as u
from astropy.constants import M_sun, R_sun, G
from luas.exoplanet import ld_to_kipping, ld_from_kipping
from luas.exoplanet import transit_2D
from luas import kernels
from luas import LuasKernel, GeneralKernel
from luas import GP
from copy import deepcopy
from corner import corner
import pymc as pm
from astropy.time import Time
from datetime import datetime
import pylightcurve as plc
from luas.pymc_ext import LuasPyMC
import pandas as pd


# This helps give more information on what PyMC is doing during inference
logging.getLogger().setLevel(logging.INFO)

# Running this at the start of the runtime ensures jax uses 64-bit floating point numbers
# as jax uses 32-bit by default
jax.config.update("jax_enable_x64", True)

# Calculates solar density in kg/m^3 to convert between fitting for stellar density or a/R*
solar_density = ((M_sun/R_sun**3)/(u.kg/u.m**3)).si
def transit_light_curve(par, t):

    # Calculates stellar density in kg/m^3 using par["a"] = a/R*
    # Can modify this function to explicitly fit for the stellar density if desired
    rho_s = 3*jnp.pi*par["a"][0]**3/(G.value*(par["P"][0]*86400)**2)

    # Creates an object describing the star
    # This code actually sets the stellar radius as 1 solar radius
    # It gives the density relative to solar density
    # This actually gives a different mass for the central star but this does not affect the transit model
    # It effectively just creates an analogous system scaled in distance by a factor R_sun/R*
    # This avoids having to input a value for R* which is irrelevant for transit calculations
    # This does not affect a/R*, Rp/R* or b as they are all dimensionless quantities
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,radius=1.)

    # Define the planetary body
    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"][0],
        time_transit=par["T0"][0],
        radius=par["rho"],
        impact_param=par["b"][0],
        eccentricity=par["e"],
        omega_peri = par["w"],
    )

    # Creates an orbit object with both the central `star` object and the `body` planet object
    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    # Define light curve function `lc` using the quadratic limb darkening coefficients u1 and u2
    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [par["u1"], par["u2"]])

    # Calculates the transit light curve flux dip with a baseline of one (default from jaxoplanet is zero)
    flux = 1 + lc(t)

    # Scales the transit model with a linear baseline
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"][0])
    
    return baseline*flux

# First we must tell JAX which parameters of the function we want to vary for each light curve
# and which we want to be shared between light curves
transit_light_curve_vmap = jax.vmap(
    # First argument is the function to vectorise
    transit_light_curve, 
    
    # Specify which parameters to share and which to vary for each light curve
    in_axes=(
        {
        # If a parameter is to be shared across each light curve then it should be set to None
        "T0":None, "P":None, "a":None, "b":None,

        # Parameters which vary in wavelength are given the dimension of the array to expand along
        # In this case we are expanding from 0D arrays to 1D arrays so this must be 0
        "rho":0, "u1":0, "u2":0, "Foot":0, "Tgrad":0
        },
        # Also must specify that we will share the time array (the second function parameter)
        # between light curves
        None,  
    ),
    
    # Specify the output dimension to expand along, this will default to 0 anyway
    # Will output extra flux values for each light curve as additional rows
    out_axes = 0,
)

def transit_light_curve_2D(p, x_l, x_t):
    
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ["T0", "P", "a", "rho", "b", "Foot", "Tgrad"]
    mfp = {k:p[k] for k in transit_params}
    
    # Calculate limb darkening coefficients from the Kipping (2013) parameterisation.
    mfp["u1"], mfp["u2"] = ld_from_kipping(p["q1"], p["q2"])
    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return transit_light_curve_vmap(mfp, x_t)

# Switch to Kipping parameterisation
if "u1" in mfp_2D:
    mfp_2D["q1"], mfp_2D["q2"] = ld_to_kipping(mfp_2D["u1"], mfp_2D["u2"])
    del mfp_2D["u1"]
    del mfp_2D["u2"]

# We implement each of these kernel functions below using the luas.kernels module
# for an implementation of the squared exponential kernel

# The wavelength kernel functions take the wavelength regression variable(s) x_l as input (of shape (N_l) or (d_l, N_l))
def Kl_fn(hp, x_l1, x_l2, wn = True):
    Kl = jnp.exp(2*hp["log_h"])*kernels.squared_exp(x_l1, x_l2, jnp.exp(hp["log_l_l"]))
    return Kl

# The time kernel functions take the time regression variable(s) x_t as input (of shape (N_t) or (d_t, N_t))
def Kt_fn(hp, x_t1, x_t2, wn = True):
    return kernels.squared_exp(x_t1, x_t2, jnp.exp(hp["log_l_t"]))

# For both the Sl and St functions we set a decomp attribute to "diag" because they produce diagonal matrices
# This speeds up the log likelihood calculations as it tells luas these matrices are easy to eigendecompose
# But don't do this for the Kl and Kt functions even if they produce diagonal matrices unless you know what you are doing
# This is because you are telling luas that the transformations of Kl and Kt are diagonal, not Kl and Kt themselves

# If the wn keyword argument is True then white noise should be included (doesn't affect most matrices)
# This is used by gp.predict when performing Gaussian process prediction
# Note it does not matter that Sl is not invertible without white noise
def Sl_fn(hp, x_l1, x_l2, wn = True):
    Sl = jnp.zeros((x_l1.shape[-1], x_l2.shape[-1]))
    
    if wn:
        # If we are including white noise then safe to assume Sl is a square matrix for any calculations in luas.GP
        Sl += jnp.diag(jnp.exp(2*hp["log_sigma"])) # Assumes hp["log_sigma"] is an array of size N_l

    return Sl
Sl_fn.decomp = "diag" # Sl is a diagonal matrix

def St_fn(p, x_t1, x_t2, wn = True):
    return jnp.eye(x_t1.shape[-1])
St_fn.decomp = "diag" # St is a diagonal matrix

# Build a LuasKernel object using these component kernel functions
# The full covariance matrix applied to the data will be K = Kl KRON Kt + Sl KRON St
kernel = LuasKernel(Kl = Kl_fn, Kt = Kt_fn, Sl = Sl_fn, St = St_fn,
                    
                    # Can select whether to use previously calculated eigendecompositions when running MCMC
                    # Performs an additional check in each step to see if each component covariance matrix has changed since last step
                    # Useful when doing blocked Gibbs or if fixing some hyperparameters
                    use_stored_values = True, 
                   )

def plot_lightcurves(x_t, M, Y, sep = 0.008):
    """Quick function to visualise light curves and the residuals after subtraction of transit model """
    N_l = x_l.shape[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12), sharey = True)
    for i in range(N_l):
        ax1.plot(x_t, Y[i, :] + np.arange(0, -sep*N_l, -sep)[i], 'bo', ms = 3)
        
        ax1.plot(x_t, M[i, :] + np.arange(0, -sep*N_l, -sep)[i], 'k-', ms = 3)
        ax2.plot(x_t, Y[i, :] - M[i, :] + np.arange(1, 1-sep*N_l, -sep)[i], 'b.', ms = 3)
        ax2.plot(x_t, np.arange(1, 1-sep*N_l, -sep)[i]*np.ones_like(x_t), 'k-', ms = 3)
    ax1.set_xlabel("Time (days)")
    ax2.set_xlabel("Time (days)")
    ax1.set_ylabel("Relative flux")
    plt.tight_layout()


def logPrior(p):
    logPrior = -0.5*((p["a"] - a_mean)/a_std)
    # logPrior += -0.5*((p["i"] - i_mean)/i_std)
    
    u1, u2 = ld_from_kipping(p["q1"], p["q2"])
    u1_priors = -0.5*((u1 - u1_mean)/u1_std)**2
    u2_priors = -0.5*((u2 - u2_mean)/u2_std)**2
    
    logPrior += u1_priors.sum() + u2_priors.sum()

    return logPrior.sum()

# Initialise our GP object
# Make sure to include the mean function and log prior function if you're using them
gp = GP(kernel,  # Kernel object to use
        x_l,     # Regression variable(s) along wavelength/vertical dimension
        x_t,     # Regression variable(s) along time/horizontal dimension
        mf = transit_light_curve_2D,  # (optional) mean function to use, defaults to zeros
        logPrior = logPrior           # (optional) log prior function, defaults to zero
       )

# Initialise our starting values as the true simulated values
p_initial = deepcopy(p_sim)

def midpoint_initval(bounds):
    lower, upper = bounds
    return np.mean([lower, upper], axis=0)

# Note with PyMC that the parameter bounds must also be NumPy arrays
param_bounds = {
                # Bounds from Kipping (2013) at just between 0 and 1
                "q1":[np.array([0]*N_l), np.array([1.]*N_l)],
                "q2":[np.array([0]*N_l), np.array([1.]*N_l)],
    
                # Can optionally include bounds on other mean function parameters
                # but often they will be well constrained by the data
                "rho":[np.array([0.]*N_l), np.array([1.]*N_l)],
                # "b":[np.array([0.]*np.ones(1)), np.array([1.]*np.ones(1))],
                "a":[np.array([0.]*np.ones(1)), np.array([20.]*np.ones(1))],
                # "T0":[np.array([-0.005]*np.ones(1)), np.array([0.005]*np.ones(1))],
                # Sometimes prior bounds on hyperparameters are important for sampling
                # However their choice can sometimes affect the results so use with caution
                "log_h":   [np.log(1e-6)*np.ones(1), np.log(1)*np.ones(1)],
                "log_l_l": [min_log_l_l*np.ones(1), max_log_l_l*np.ones(1)],
                "log_l_t": [min_log_l_t*np.ones(1), max_log_l_t*np.ones(1)],
                "log_sigma":[np.log(1e-6)*np.ones(N_l), np.log(1e-2)*np.ones(N_l)],
}

# Make a wrapper function which returns a PyMC model with a given set of fixed parameters and observations Y
def transit_model(p_fixed, Y): 
    
    with pm.Model() as model:

        # Makes of copy of any parameters to be kept fixed during sampling
        var_dict = deepcopy(p_fixed)
        
        # Specify the parameters we've given bounds for
        var_dict["rho"] = pm.Uniform('rho', lower=param_bounds["rho"][0],
                                     upper=param_bounds["rho"][1], shape=N_l, initval=midpoint_initval(param_bounds["rho"]))
        var_dict["q1"] = pm.Uniform('q1', lower=param_bounds["q1"][0],
                                    upper=param_bounds["q1"][1], shape=N_l, initval=midpoint_initval(param_bounds["q1"]))
        var_dict["q2"] = pm.Uniform('q2', lower=param_bounds["q2"][0],
                                    upper=param_bounds["q2"][1], shape=N_l, initval=midpoint_initval(param_bounds["q2"]))
        var_dict["log_h"] =   pm.Uniform("log_h", lower=param_bounds["log_h"][0],
                                         upper=param_bounds["log_h"][1], shape=1, initval=midpoint_initval(param_bounds["log_h"]))
        var_dict["log_l_l"] = pm.Uniform("log_l_l", lower = param_bounds["log_l_l"][0],
                                         upper = param_bounds["log_l_l"][1], shape=1, initval=midpoint_initval(param_bounds["log_l_l"]))
        var_dict["log_l_t"] = pm.Uniform("log_l_t", lower = param_bounds["log_l_t"][0],
                                         upper = param_bounds["log_l_t"][1], shape=1, initval=midpoint_initval(param_bounds["log_l_t"]))
        var_dict["log_sigma"] = pm.Uniform('log_sigma', lower=param_bounds["log_sigma"][0],
                                           upper=param_bounds["log_sigma"][1], shape=N_l, initval=midpoint_initval(param_bounds["log_sigma"]))
        var_dict["a"] = pm.Uniform('a', lower=param_bounds["a"][0],
                                     upper=param_bounds["a"][1], shape=1, initval=midpoint_initval(param_bounds["a"]))

        # Specify the unbounded parameters
        var_dict["T0"] = pm.Flat('T0', shape=1)
        var_dict["b"] = pm.Flat('b', shape=1)
        var_dict["Foot"] = pm.Flat('Foot', shape=N_l)
        var_dict["Tgrad"] = pm.Flat('Tgrad', shape=N_l)

        # PyMC wrapper for luas log posterior calculations
        # Requires the gp object, a dictionary of each model parameter and the observations Y
        LuasPyMC("log_like", gp = gp, var_dict = var_dict, Y = Y)
        
    # Will need to return both the model and the model variables for inference
    return model, var_dict

# Initialise our model, p_initial will specify any fixed values like the period P
model, var_dict = transit_model(p_initial, Y)

# PyMC requires the dictionary of starting values to only include variables in the model
# So we must remove the period parameter P as we keep it fixed
p_pymc = deepcopy(p_initial)
del p_pymc["P"]

# Use PyMC's maximum posteriori optimisation function
map_estimate = pm.find_MAP(
    model = model,                  # PyMC model to optimise
    include_transformed = False,    # If this is true it will also output the PyMC transformed values of bounded parameters
    start = p_pymc,                 # Starting point of optimisations
    maxeval = 5000,                 # Maximum steps to run (normally will converge before this)
)

# Create a new dictionary of optimised values which includes our fixed parameters
p_opt = deepcopy(p_initial)
p_opt.update(map_estimate)

print("Starting log posterior value:", gp.logP(p_initial, Y))
print("New optimised log posterior value:", gp.logP(p_opt, Y))

# This function will return a JAXArray of the same shape as Y
# but with outliers replaced with interpolated values
Y_clean = gp.sigma_clip(p_opt, # Make sure to perform sigma clipping using a good fit to the data
                        Y,     # Observations JAXArray
                        5.     # Significance level in standard deviations to clip at
                       )

# Returns the covariance matrix returned by the Laplace approximation
# Also returns a list of parameters which is the order the array is in
# This matches the way jax.flatten_util.ravel_pytree will sort the parameter PyTree into
cov_mat, ordered_param_list,  = gp.laplace_approx_with_bounds(
    p_opt,               # Make sure to use best-fit values
    Y_clean,             # The observations being fit
    param_bounds,        # Specify the same bounds that will be used for the MCMC
    fixed_vars = ["P"],  # Make sure to specify fixed parameters as otherwise they are marginalised over
    return_array = True, # May optionally return a nested PyTree if set to False which can be more readable
    regularise = True,   # Often necessary to regularise values that return negative covariance
    large = False,       # Setting this to True is more memory efficient which may be needed for large data sets
)

# This function will output information on what regularisation has been performed
# And will mention if there are remaining negative values along the diagonal of the covariance matrix
# It does not however check if the covariance matrix is invertible

# Initialise our PyMC model
model, var_dict = transit_model(p_opt, Y_clean)

# The NUTS step takes as input the variables created when initialising the model
# We also sort these variables in the same order our Laplace approximated covariance matrix is in
NUTS_model_vars = [var_dict[par] for par in ordered_param_list]
NUTS_step = pm.NUTS(NUTS_model_vars, scaling = cov_mat, is_cov = True, model = model)

# Begin MCMC sampling
idata = pm.sample(
    model = model,           # PyMC model to sample
    step = NUTS_step,        # Sampling steps to use (can be list for blocked Gibbs sampling)
    initvals = map_estimate, # Starting point of inference (will jitter around this location)
    draws = 1000,            # Number of samples from MCMC post warm-up
    tune = 1000,             # Number of tuning steps
    chains = 4,              # Number of chains to run
    cores = 1,               # This will probably fail if not set to 1 as I don't think PyMC can parallelise JAX functions
)

# Saves the inference object
idata.to_json("MCMC_chains.json")

az.summary(idata, round_to = 4)

trace_plot = az.plot_trace(idata)
plt.tight_layout()

# Select the parameters to include in the plot
params = ["T0", "a", "rho", "b", "log_h", "log_l_l", "log_l_t"]

# Plot each of the two chains separately
idata_corner1 = idata.sel(chain=[0])
idata_corner2 = idata.sel(chain=[1])
idata_corner3 = idata.sel(chain=[2])
idata_corner4 = idata.sel(chain=[3])

# Plot first chain
fig1 = corner(idata_corner1, smooth = 0.4, var_names = params)

# Plot second chain along with truth values
fig2 = corner(idata_corner1, quantiles=[0.16, 0.5, 0.84], title_fmt = None, title_kwargs={"fontsize": 23},
              label_kwargs={"fontsize": 23}, show_titles=True, smooth = 0.4, color = "r", fig = fig1,
              top_ticks = True, max_n_ticks = 2, labelpad = 0.16, var_names = params,
              truths = p_sim, truth_color = "k",
              )

# NumPy array of MCMC samples of shape (N_chains, N_draws, N_l)
rho_chains = idata.posterior["rho"].to_numpy()

# Get the mean radius ratio values averaged over all chains and draws
rho_mean = rho_chains.mean((0, 1))

# Calculate the covariance matrix of each chain and average together
N_chains = rho_chains.shape[0]
rho_cov = jnp.zeros((N_l, N_l))
for i in range(N_chains):
    rho_cov += jnp.cov(rho_chains[0, :, :].T)
rho_cov /= N_chains

# Standard deviation given by sqrt of diagonal of covariance matrix
rho_std_dev = jnp.sqrt(jnp.diag(rho_cov))

# We can plot our recovered spectrum against the simulated true spectrum
plt.errorbar(x_l, rho_mean, yerr = rho_std_dev/3, fmt = 'k.-', label = "Recovered Spectrum")
plt.plot(x_l, p_sim["rho"], 'r--', label = "True Spectrum")

# Select a few samples from the MCMC to help visualise the correlation between values
rho_draws = rho_chains[0, 0:1000:100, :].T
# plt.plot(x_l, rho_draws, 'k-', alpha = 0.3)
plt.xlabel(r"Wavelength ($\AA$)")
plt.ylabel(r"Radius Ratio $\rho$")
plt.legend()
plt.show()

plt.title("Tranmission spectrum covariance matrix")
plt.imshow(rho_cov, extent = [x_l[0], x_l[-1], x_l[-1], x_l[0]])
plt.colorbar()
plt.xlabel(r"Wavelength ($\AA$)")
plt.ylabel(r"Wavelength ($\AA$)")
plt.show()