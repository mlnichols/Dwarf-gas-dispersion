"""
Calculates the alpha metallicity dispersion in the gas for a GEAR based simulation.
This relies on the pNbody package. See http://obswww.unige.ch/~revaz/pNbody/
Although the fundamentals of the code could easily be extend to deal with any other SPH like simulation.
"""

import numpy as np
import pNbody
import sph_convolution
import matplotlib.pyplot as plt

#Need a mass weighted standard deviation, since more massive regions = more likely to produce stars
def weight_std(vals,weights):
    mean = np.average(vals,weights=weights)
    var = np.average((vals-mean)*(vals-mean),weights=weights)
    return np.sqrt(var)

#Use a generic snapshot here.
#In practice this would be looped over on multiple cores for the snapshots and the final results input into database to examine the parameter space
nb = pNbody.Nbody('snapshot',ftype='gadget')

#Select only the gas particles, since that is what we care about
#These gas particles are lagrangian units, representing the centre of a mass element, however, the value of any property is defined by a convolution over them! Using just the value of said mass element would increase dispersion.
gas = nb.select(0)

#The limits are given by the maximum of the value in each dimension plus the smoothing length and min-minus smoothing length.
limmax = np.max(gas.pos+np.array([gas.rsp,gas.rsp,gas.rsp]).transpose(),axis=0)
limmin = np.min(gas.pos-np.array([gas.rsp,gas.rsp,gas.rsp]).transpose(),axis=0)

#By default these are in units of kpc, now star forming regions are on average 1.5 kpc in size. So using 1 kpc pixels makes sense.
#Now it only makes sense to use the ceiling and floor since otherwise the edge won't have a value
lims = np.array([np.ceil(limmin),np.floor(limmax)]).transpose()
npix = (lims[:,1]-lims[:,0])

#Metals content
Fe = gas.metals[:,0] #gas.metals contains the fractional mass of gas
Mg = gas.metals[:,1]

#Solar abundances
Solar = [0.001771,0.00091245]

#Calculate the metal content across the galaxy
#To a simple good approximation in the static case the SPH volumes are equivalent between the density and the entropy based simulations
#NOTE: We are really computing the density of metals in each pixel! So we also need a base density, Hydrogen+Helium is in these galaxies nearly everything, so just use that
Fe_gal = sph_convolution.grid(gas.pos.astype(np.float),gas.mass.astype(np.float)/gas.Rho().astype(np.float),gas.rsp.astype(np.float),Fe.astype(np.float)*gas.Rho().astype(np.float),lims.astype(np.float),npix.astype(np.int))
Mg_gal = sph_convolution.grid(gas.pos.astype(np.float),gas.mass.astype(np.float)/gas.Rho().astype(np.float),gas.rsp.astype(np.float),Mg.astype(np.float)*gas.Rho().astype(np.float),lims.astype(np.float),npix.astype(np.int))
Dens_gal = sph_convolution.grid(gas.pos.astype(np.float),gas.mass.astype(np.float)/gas.Rho().astype(np.float),gas.rsp.astype(np.float),gas.Rho().astype(np.float),lims.astype(np.float),npix.astype(np.int))

#Conversion to [Fe/H], in both cases hydrogen is so close to 1 we can approximation
Fe_prop = np.log10(Fe_gal/Dens_gal) - np.log10(Solar[0])
MgFe_prop = np.log10(Mg_gal/Fe_gal) - np.log10(Solar[1]/Solar[0])

#Now the dispersion is only important above where stars form, before we run into Type Ia Supernovae issues
inds = (Fe_prop > -5)  #Get all values in this range. np.nan evaluates to false in both so that's fine

Dens_gal *= 407.617 #Convert to physical units particles/cm^3
inds *= Dens_gal > 1e-2 #Only consider the areas stars can actually form in the future, i.e. those regions dense enough

#This could now be outputted
deviation = weight_std(MgFe_prop[inds],weights=Dens_gal[inds])
print 'Deviation of [Mg/Fe] at the metallicity plateau is %6.4f this should be <0.1 or it doesn\'t match observations!' % deviation
#Alternatively plot it
plt.scatter(Fe_prop[inds],MgFe_prop[inds],c=Dens_gal[inds],alpha=0.3)
plt.show()