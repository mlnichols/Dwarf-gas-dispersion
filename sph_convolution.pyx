import numpy as np
cimport numpy as np
DTYPE = np.float
ITYPE = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t ITYPE_t

#Generate a grid of points to calculate the dispersion over. As star formation happens on characteristic scales, this is normally fine.
#The fundamental idea here is easy: For each particle, calculate which pixels it contributes to and sum it up. Then divide by the arrays at the end.
#With the scale of each individual simulation snapshot it isn't worth the time parallelizing, however, outputting valsum and sphsum and then summing with MPI_SUM and dividing the result will give full parallelization
def grid(np.ndarray[DTYPE_t, ndim=2] pos, np.ndarray[DTYPE_t, ndim=1] volume, np.ndarray[DTYPE_t, ndim=1] smooth, np.ndarray[DTYPE_t, ndim=1] val,  np.ndarray[DTYPE_t, ndim=2] lims, np.ndarray[ITYPE_t,ndim=1] points):
    """
    grid computes the SPH convolution over a grid of supplied points
    assuming a cubic spline kernel.

    Parameters
    ----------
    pos : Array of shape (n,3)
        n by three dimenisonal array which contains the position of each
        of the n particle in the three dimensions
    volume : Array
        One dimensional array of the \'volume\' of each particle where
        the i\'th element corresponds to the i\'th element in pos. Where
        the \'volume\' is defined as per the SPH scheme.
    smooth : Array
        One dimensional array of the smoothing lengths of the particles
    val : Array
        The relevant value to compute the SPH value at the grids.
    lims : Array of shape (3,2)
        Limits of the grid
    points : Array of size 3
        Number of points along each dimension between the limits of lims.

    Returns
    -------
    array of size(points[1],points[0],points[2])
        Array of the SPH computed value at each point. Arranged so that
        the y-values decrease as you increase the row number. This is
        done for graphical compatability with imshow of pyplot.    
    """
    cdef int nmax = pos.shape[0]

    #im is the output array, it is an image of the SPH values at each point
    cdef np.ndarray[DTYPE_t, ndim=3] im = np.zeros([points[1],points[0],points[2]],dtype=DTYPE)
    #sphsum is a running array which computes the density in the SPH scheme, as this is a linear sum we don't need to normalize till the end
    cdef np.ndarray[DTYPE_t, ndim=3] sphsum = np.zeros([points[1],points[0],points[2]],dtype=DTYPE)
    #valsum does the same for the value
    cdef np.ndarray[DTYPE_t, ndim=3] valsum = np.zeros([points[1],points[0],points[2]],dtype=DTYPE)

    #Counters, indicies, pixel sizes, differentials, kernel values and smoothing length distances
    cdef int i,j,k,n
    cdef ITYPE_t ib,jb,kb,i0,j0,k0,i1,j1,k1
    cdef unsigned int N,I,J,K,JMAX
    cdef float resx, resy, resz
    cdef DTYPE_t h,h2,hinv,hinv3,KC1,KC2,KC5,px,py,pz,dx,dy,dz,wk,r2,r,u,t1
    cdef DTYPE_t rmax = max(smooth)

    #Value of each pixel in x, y, and z
    px =  abs(lims[0,0]-lims[0,1])/points[0]
    py =  abs(lims[1,0]-lims[1,1])/points[1]
    pz =  abs(lims[2,0]-lims[2,1])/points[2]
    #As we want y=0 to be the highest row we need the maximum value
    JMAX = <unsigned int>(points[1]-1)
    #Kernel values, they're in reality functions of pi, but no need for full precision
    KC1 = 2.546479089470
    KC2 = 15.278874536822
    KC5 = 5.092958178941

    #loop over the pixels
    for n in range(0,nmax):
        N = <unsigned int>(n) #Slightly quicker to use unsigned ints for indicies in cython
        #Smoothing length parameters
        h = smooth[N] 
        h2 = h*h
        hinv = 1/h
        hinv3 = hinv*hinv*hinv
        
        #Calculating the position of the particle, this will comprise integer of array + residual
        resx = (pos[N,0] - lims[0,0])/px 
        resy = (pos[N,1] - lims[1,0])/py 
        resz = (pos[N,2] - lims[2,0])/pz 

        #Extract the integer or pixel array position, using standard i,j,k notation for x,y,z
        ib = np.floor(resx)
        jb = np.floor(resy) 
        kb = np.floor(resz)

        #Calculate residual
        resx = resx%1
        resy = resy%1
        resz = resz%1

        #Calculate the limits of the pixels it can impact.
        #Only these need to be looped over!
        i0 = np.int(min(max(ib-h/px-1,0),points[0]-1))
        j0 = np.int(min(max(jb-h/py-1,0),points[1]-1))
        k0 = np.int(min(max(kb-h/pz-1,0),points[2]-1))

        i1 = np.int(max(min(ib+h/px+1,points[0]),1))
        j1 = np.int(max(min(jb+h/py+1,points[1]),1))
        k1 = np.int(max(min(kb+h/pz+1,points[2]),1))

        #Now do a nested loop for the three dimensions to loop over every pixel
        #Nominally this is something that could be vectorized in normal python, but anecdotally its faster in cython.
        for i in range(i0,i1):
            I = <unsigned int>(i)
            dx = i*px+lims[0,0] - pos[N,0]
            for j in range(j0,j1):
                J = <unsigned int>(j)
                dy = j*py+lims[1,0] - pos[N,1]
                for k in range(k0,k1):
                    K= <unsigned int>(k)
                    dz = k*pz+lims[2,0] - pos[N,2]
                    
                    r2 = dx*dx + dy*dy + dz*dz
                    
                    #This is the actual kernel value calculation
                    if(r2 < h2):
                        r = r2**0.5
                        u = r*hinv
                        t1 = (1.0-u)
                        if(u<0.5):
                            wk = hinv3 * (KC1+KC2*(u-1)*u*u)
                        else:
                            wk = hinv3 * KC5 * t1 * t1 * t1
                        sphterm = volume[N] * wk
                        sphsum[JMAX-J,I,K] += sphterm
                        valsum[JMAX-J,I,K] += sphterm*val[N]


    #Now divide the arrays but ensure that the values where there are no particles don't give weird results like infinity! Since they all start at 0.
    #An alternative is to output both valsum and sphsum here, and divide them very quickly in python
    for i in range(0,points[0]):
        I = <unsigned int>(i)
        for j in range(0,points[1]):
            J = <unsigned int>(j)
            for k in range(0,points[2]):
                K = <unsigned int>(k)
                if(sphsum[J,I,K] != 0):
                    im[J,I,K] = valsum[J,I,K]/sphsum[J,I,K]
                else:
                    im[J,I,K] = np.nan

        
    return im