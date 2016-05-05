import numpy as np
from math import pi,sin,cos,sqrt,atan,acos
#from scipy import misc 
from sys import argv
from mpi4py import MPI
from time import time, ctime

a=np.array([0, .2, .3, .6, 1., 7./8.])
b=np.array([[0.,0.,0.,0.,0.],[.2,0.,0.,0.,0.],[3./40.,9./40.,0.,0.,0.],[3./10., -9./10., 6./5., 0., 0.],[-11./54., 2.5, -70./27., 35./27., 0.], [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
c = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
cstar = np.array([2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 0.25])
dc = np.array([277./64512.,0.,-6925./370944.,6925./202752.,277./14336.,-277./7084.])
lena=len(a)



# t is lamb, affine parameter
# y is u, four velocity, or x, four position

def initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,imagewidth, imageheight, Rs):
    #set origin of pixel plane at x axis
    t = 0.
    x = Rplane
    y = (pixelcoord[0]-(pixelwidth-1)/2.)*imagewidth/float(pixelwidth)
    z = (pixelcoord[1]-(pixelheight-1)/2.)*imageheight/float(pixelheight)
    r= sqrt(x*x+y*y+z*z)
    phi = atan(y/x)
    theta = acos(z/r)

    #initial u perpendicular to plane.
    #magnitude of u is arbitrary-- affine parameter makes it rescalable
    #(ut)^2-(uy)^2-(ux)^2-(uz)^2=0 so ut = +-ux
    #for x decreasing as t increases, ut = -ux (inward)
    uy = 0.
    uz = 0.
    ux = 1.


    invr = 1./r
    invrsq = invr*invr
    ur = -x/r
    utheta = x*z*invr*invrsq*sqrt(1.-z*z*invrsq)
    uphi = -y/(x*x+y*y)
    rmRs = r-Rs
    st = sin(theta)
    ut = sqrt((ur*ur*r/rmRs+r*r*utheta*utheta+r*r*st*st*uphi*uphi)/rmRs*r)
    initcoords =np.array([t, r, theta, phi, ut, ur, utheta, uphi])
    return initcoords

def initializeElliptical(eccentricity,semilatusr,Rs):
    r2 = semilatusr*0.5*Rs/(1.-eccentricity)
    print(eccentricity, semilatusr, Rs,r2)
    theta = pi/2.
    phi = 0.
    t = 0.
    utheta = 0.
    temp = 1./(semilatusr - 3. - eccentricity*eccentricity)
    angularL = 0.5*semilatusr*Rs*sqrt(temp)
    energy=sqrt((semilatusr-2.-2.*eccentricity)*(semilatusr-2.+2.*eccentricity)/semilatusr*temp)
    uphi = angularL/r2/r2
    ur = 0.
    ut = energy/(1.-Rs/r2)
    return np.array([t,r2,theta,phi,ut,ur,utheta,uphi])

def adaptiveRK4(t,y,h,func,maxfunc,arg,yscale,epsilon):
    leny=len(y)
    safetyfac = 0.9
    pgrow =-0.20
    pshrink =-0.25
    errcon = 1.89e-4 #see NR in Fortran
    hnew=h/2.
    nsteps=0
    #hlast = h
    while True:
        # j and i are reversed from Numerical Recipes book (page 711)
        #loop over y indices
        k=np.zeros((leny,lena))
        tprimearg=t
        yprime = np.copy(y)
        yprimestar = np.copy(y)
        for j in range(0,len(a)): #for all terms summed in method
            tprimearg = t+a[j]*h
            yprimearg = np.copy(y)
            for n in range(0,leny): #over all variables in y vector
                for i in range(0,j): #over all indices of k
                    #update for next term of k in calculation
                    yprimearg[n]+=b[j,i]*k[n,i]
            k[:,j]=h*func(tprimearg,yprimearg,arg)
        yprime = y+np.sum(np.multiply(c,k),axis=1)
        yerr =  np.sum(np.multiply(dc,k),axis=1)
        yprimestar =np.copy(y)+ np.sum(np.multiply(cstar,k),axis=1)
        errmax = maxfunc(yerr,yscale,yprime)
        errmax/=epsilon
        if (errmax>1):
            hnew = safetyfac*h*pow(errmax,pshrink)
            if(hnew<0.1*h):
                hnew=.1*h
            h=hnew
            nsteps+=1
        else:
            if(errmax>errcon):
                hnew = safetyfac*h*pow(errmax,pgrow)
            else:
                hnew = 5.*h
            nsteps+=1
            return nsteps,t+h,yprimestar,hnew
    tprime = t+h
    return tprime,yprimestar,hnew


def linearMaxFunc(yerr,yscale,yprime):
    errmax = max(np.absolute(yerr/yscale))
    return errmax



def geodesic(lamb,x,Rs):
    #returns a vector of the four acceleration
    #declare some constants
    rmRs=x[1]-Rs
    ct=cos(x[2])
    st=sin(x[2])
    invrmRs = 1./rmRs
    invr = 1./x[1]
    temp1 = 0.5*Rs*invr
    x7sq = x[7]*x[7]
    x5invr=x[5]*invr
    #calculate dut
    dut = -Rs*x[4]*x[5]*invr*invrmRs
    #calculate dur
    cut = -temp1*rmRs*invr*invr
    cur = temp1*invrmRs
    cutheta = rmRs
    cuphi = rmRs*st*st
    #print("x=",x)
    dur =cut*x[4]*x[4]+cur*x[5]*x[5]+cutheta*x[6]*x[6]+cuphi*x7sq
    #calculate dutheta
    dutheta = -2.*x[6]*x5invr+ct*st*x7sq
    #calculate duphi
    duphi =-2.*(x5invr+ct/st*x[6])*x[7]
    rhs=np.array([x[4],x[5],x[6],x[7],dut, dur, dutheta, duphi])
    #print(cut,cur,cutheta,cuphi)
    return np.array([x[4],x[5],x[6],x[7],dut, dur, dutheta, duphi])

def integrateNullGeodesic(xpix, ypix, pixelheight,pixelwidth, skypixelheight,skypixelwidth,imagewidth,imageheight,Rs,Router,Rplane,eccentricity, semilatusr, epsilon, tiny, hinit,Rfac,heps):
    pixelcoord=np.array([xpix,ypix])
    coords = initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,imagewidth,imageheight,Rs).copy()
    r=coords[1]
    lamb=0.
    color = 1
    n=0
    h=hinit
    phi=coords[3]
    totnsteps=0
    while(r<=Router):
        yscale =np.absolute(coords)+np.absolute(h*geodesic(lamb,coords,Rs))+tiny
        nsteps,lamb,coords,h=adaptiveRK4(lamb,coords,h,geodesic,linearMaxFunc,Rs,yscale,epsilon)
        r=coords[1]
        phi=coords[3]
        totnsteps+=nsteps
        if (r<Rfac*Rs) and (h<heps):
            color = 0
            break
        n+=1
        if((n%10000)==0): print(n,r,coords[2],phi,h)
    if(coords[2]<0.):
        temp = (-coords[2])%(pi)
        coords[2]=pi-temp
    else:
        coords[2]%=pi
                
    if(coords[3]<0.):
        temp=(-coords[3])%(2.*pi)
        coords[3]=2.*pi-temp
    else:
        coords[3]%=(2.*pi)
    rmRs2 = coords[1]-Rs
    telestart = (xpix+ypix*pixelwidth)*3
    xout = int(coords[3]*skypixelwidth/2./pi)
    yout = int(coords[2]*skypixelheight/pi)
    skystart = (xout+yout *skypixelwidth)*3
    return totnsteps,skystart,telestart,color

    
def main():

    comm = MPI.COMM_WORLD
    id= comm.Get_rank()
    wsize= comm.Get_size()    
    tstart = MPI.Wtime()
    #fsky = open("skymap.png","r")
    #reader = Reader(fsky)
    #skypixelwidth, skypixelheight, skypixels, metadata=reader.read_flat()
    skypixelwidth = 4096
    skypixelheight = 2048
    skypixels = np.zeros((skypixelwidth*skypixelheight*3),dtype=np.uint8)
    pixelwidth = int(argv[1])
    pixelheight = int(argv[2])
    tskymapstart = MPI.Wtime()
    telepixels = np.zeros((pixelwidth*pixelheight*3),dtype=np.uint8)
    colorpixels = np.zeros((pixelwidth*pixelheight),dtype=np.uint8)
    skystartall = np.zeros((pixelwidth*pixelheight),dtype=np.uint32)
    telestartall = np.zeros((pixelwidth*pixelheight),dtype=np.uint32)
    colorall = np.zeros((pixelwidth*pixelheight),dtype=np.uint8)
    totnstepsall=np.zeros((wsize),dtype=np.uint32)
    tskymapend = MPI.Wtime()
    tskymap = tskymapend-tskymapstart

    tmin = 1.e6
    tpercparmin=1.e6
    hinit=1.e-1
    #h=1.e-4
    Router = 1000.
    Rplane = 700.
    Rs = 2.
    every = 1
    deltalamb = 1.e-1
    imagewidth = 50.;
    imageheight = 50.;
    tiny = 1.e-30
    epsilon=1.e-8
    eccentricity = 0.2
    Rfac = 1.+1.e-10
    heps = 1.e-14
    semilatusr = 10.0  

    
    tstartpp=MPI.Wtime() #percent parallelized
    numperprocess = pixelheight*pixelwidth/wsize
    skystart=np.zeros((numperprocess),dtype=np.int32)
    telestart=np.zeros((numperprocess),dtype=np.int32)
    color = np.zeros((numperprocess),dtype=np.int8)
    totnsteps=np.zeros((numperprocess),dtype=np.int32)
    trk4all=np.zeros((numperprocess),dtype=np.float)
    ttelestop = MPI.Wtime()
    ttele = ttelestop-tstartpp
    trk4=float("inf")
    for index in range(numperprocess):
        ypix = int((id*numperprocess+index)/pixelwidth)
        xpix = (id*numperprocess+index)%pixelwidth
        tstartrk4=MPI.Wtime()
        totnsteps[index],skystart[index],telestart[index],color[index]=integrateNullGeodesic(xpix, ypix, pixelheight,pixelwidth, skypixelheight,skypixelwidth,imagewidth,imageheight,Rs,Router,Rplane,eccentricity, semilatusr, epsilon, tiny, hinit,Rfac,heps)
        tendrk4=MPI.Wtime()
        trk4=min(trk4,(tendrk4-tstartrk4)/float(totnsteps[index]))
    totnstepsmax=max(totnsteps)
    tstoppp = MPI.Wtime()
    tpercpar=tstoppp-tstartpp

    comm.Barrier()
    if id==0:
        totnstepsmaxall=0
    else:
        totnstepsmaxall=None
    comm.Barrier()

    totnstepsmaxall=comm.reduce(totnstepsmax,op=MPI.MAX,root=0)
    tskymapall = comm.reduce(tskymap, op=MPI.MAX, root=0)
    tteleall = comm.reduce(ttele,op=MPI.MAX,root=0)
    comm.Gatherv(skystart,skystartall,root=0)
    comm.Gatherv(telestart, telestartall, root=0)
    comm.Gatherv(color,colorall, root=0)
    trk4min=comm.reduce(trk4,op=MPI.MIN,root=0)
    comm.Barrier()
    tend = MPI.Wtime()
    tall = tend-tstart
    if id==0:
        tindexstart = MPI.Wtime()
        for index in range(pixelheight*pixelwidth):

            if(colorall[index]==1):
                telepixels[telestartall[index]:telestartall[index]+3]=skypixels[skystartall[index]:skystartall[index]+3]
            else:
                telepixels[telestartall[index]]=255 #leave other two indices zero,red
        tindexend = MPI.Wtime()
        tindex = tindexend-tindexstart
    #if id==0:
        #twritestart = MPI.Wtime()
        #ftele = open('teleview_{pw}_{ph}_{ws}.png'.format(pw=pixelwidth,ph=pixelheight,ws=wsize), "w")
        #telewrite=Writer(width=pixelwidth,height=pixelheight,greyscale=False,alpha=False)
        #telewrite.write_array(ftele,telepixels)
        #ftele.close()
        #twriteend=MPI.Wtime()
        #twrite = twriteend-twritestart
    #fsky.close()
    comm.Barrier()
    tmax = comm.reduce(tall,MPI.MAX,root=0)
    tpercparmin = comm.reduce(tpercpar/tall,op=MPI.MIN,root=0)
    comm.Barrier()
    if (id==0):
#        print("Telescope dimensions in M", 2.*imagewidth, 2.*imageheight)
#        print("Telescope resolution", pixelwidth, pixelheight)
#        print("Skymap resolution", skypixelwidth, skypixelheight)
#        print("Schwarzschild radius in M", 2.*Rs)
#        print("Outer radius in M", 2.*Router)
#        print("Telescope radius in M", 2.*Rplane)
#        print("Number of processes = ",wsize)
#        print("Maximum number of integration steps taken is",totnstepsmaxall)
#        print("The time for a single step of the RK4 is",trk4min)
#        print("Total runtime = ",tmax)
#        print("Fraction parallel = ", tpercparmin)
        print pixelwidth,pixelheight,wsize,totnstepsmaxall,trk4min,tmax,tpercparmin, tindex, tskymapall, tteleall

    MPI.Finalize()
main()

