import numpy as np
from math import pi,sin,cos,sqrt,atan,acos
from matplotlib import pyplot

# t is lamb, affine parameter
# y is u, four velocity, or x, four position

def initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,imagewidth, imageheight, Rs):
    #set origin of pixel plane at x axis
    t = 0.
    x = Rplane
    y = (pixelcoord[0]-pixelwidth/2.)*imagewidth/float(pixelwidth)
    z = (pixelcoord[1]-pixelheight/2.)*imageheight/float(pixelheight)
    r = sqrt(x*x+y*y+z*z)
    phi = atan(y/x)
    theta = atan(sqrt(x*x+y*y)/r)

    #initial u perpendicular to plane.
    #magnitude of u is arbitrary-- affine parameter makes it rescalable
    #(ut)^2-(uy)^2-(ux)^2-(uz)^2=0 so ut = +-ux
    #for x decreasing as t increases, ut = -ux (inward)
    uy = 0.
    uz = 0.
    ux = 1.
    rhosq = x*x + y*y
    facrrho= rhosq+r*r
    #for this specific case, where ux = 1:
    ur = -x/r
    utheta = -x*z*z/sqrt(rhosq)/r/facrrho
    uphi = y/rhosq
    rmRs = r-Rs
    st = sin(theta)
    ut = sqrt((ur*ur*r/rmRs+r*r*utheta*utheta+r*r*st*st*uphi*uphi)/rmRs*r)
    initcoords =np.array([t, r, theta, phi, ut, ur, utheta, uphi])
#    coords = initcoords
#    rmRs2 = coords[1]-Rs
#    testnull = -rmRs2/coords[1]*coords[4]*coords[4]+coords[5]*coords[5]*coords[1]/rmRs2+coords[1]*coords[1]*(coords[6]*coords[6]+sin(coords[2])*sin(coords[2])*coords[7]*coords[7])
    testnull = -rmRs/r*ut*ut+ur*ur*r/rmRs+r*r*(utheta*utheta+st*st*uphi*uphi)
    print(testnull)
    print(testnull/max(np.absolute(initcoords[4:8])))
    return initcoords

def initializeElliptical(eccentricity,semilatusr,Rs):
    r2 = semilatusr*0.5*Rs/(1.-eccentricity)
    theta = pi/2.
    phi = 0.
    t = 0.
    utheta = 0.
    temp = 1./(semilatusr - 3. - eccentricity*eccentricity)
    angularL = 0.5*semilatusr*Rs*sqrt(temp)
    energy=sqrt((semilatusr-2.-2.*eccentricity)*(semilatusr-2.+2.*eccentricity)/semilatusr*temp)
    uphi = angularL/r2/r2
    ur = 0.
    ut = energy/sqrt(1.-Rs/r2)
    return np.array([t,r2,theta,phi,ut,ur,utheta,uphi])

def adaptiveRK4(t,y,h,func,maxfunc,arg,yscale,epsilon):
    a=np.array([0, .2, .3, .6, 1., 7./8.])
    b=np.array([[0.,0.,0.,0.,0.],[.2,0.,0.,0.,0.],[3./40.,9./40.,0.,0.,0.],[3./10., -9./10., 6./5., 0., 0.],[-11./54., 2.5, -70./27., 35./27., 0.], [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
    c = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
    cstar = np.array([2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 0.25])
    dc = np.array([277./64512.,0.,-6925./370944.,6925./202752.,277./14336.,-277./7084.])
    safetyfac = 0.9
    pgrow = 0.20
    pshrink =0.25
    errcon = 1.89e-4 #see NR in Fortran
    hnew=h/2.
    lena=len(a)
    leny=len(y)
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
        #delta0 = np.absolute(np.multiply(epsilon,yscale))
        #yerr = yprime - yprimestar
        errmax = maxfunc(yerr,yscale,yprime)
        errmax/=epsilon
        if (errmax>1):
            hnew = safetyfac*h*pow(errmax,pshrink)
            if(hnew<0.1*h):
                hnew=.1*h
            h=hnew
        else:
            if(errmax>errcon):
                hnew = safetyfac*h*pow(errmax,pgrow)
            else:
                hnew = 5.*h
            return t+h,yprimestar,hnew
            #false break for testing
            #break    
            #problem has something to do with break conditions
            #print("breaking")
    #tprime = t+h
    #print(h)
    tprime = t+h
    return tprime,yprimestar,hnew
    #return tprime,yprimestar,h

def linearMaxFunc(yerr,yscale,yprime):
    errmax = max(np.absolute(yerr/yscale))
    return errmax

def sphericalMaxFunc(yerr,yscale,x):
    rscale=yscale[1]/x[1]
    invst = 1./sin(x[2])
    errmax = max(abs(x[0]/yscale[0]),abs(x[1]/yscale[1]),abs(x[2]/yscale[2]*rscale),abs(x[3]/yscale[3]*invst*rscale),abs(x[4]/yscale[4]),abs(x[5]/yscale[5]),abs(x[6]/yscale[6]*rscale),abs(x[7]/yscale[7]*rscale*invst))
    return errmax


def rk4(t,y,h,func,arg):
    k1=h*func(t,y,arg) #no t on right hand side of these equations
    k2 = h*func(t+0.5*h,y+0.5*k1,arg)
    k3 = h*func(t+0.5*h,y+0.5*k2,arg)
    k4 = h*func(t+h,y+k3,arg)
    t=t+h
    return t,np.copy(y)+k1/6.+k2/3.+k3/3.+k4/6.

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
    dutheta = -2.*x[6]*x5invr++ct*st*x7sq
    #calculate duphi
    duphi =-2.*(x5invr+ct/st*x[6])*x[7]
    rhs=np.array([x[4],x[5],x[6],x[7],dut, dur, dutheta, duphi])
    #print(cut,cur,cutheta,cuphi)
    #print("rhs=",rhs)
    return np.array([x[4],x[5],x[6],x[7],dut, dur, dutheta, duphi])

#parabola test
def parabola(t,y,ab):
    ab[0] = a
    ab[1] =b
    ynew=np.array([a*t+b,a*t+b])
    return ynew

def sinusoid(t,y,params):
    amp = params[0]
    omega = params[1]
    phase = params[2]
    ynew = amp*omega*cos(omega*t+phase)
    ynewarray = np.array([ynew,ynew])
    return ynewarray

def sho(t,y,omega): #y[0] is u, y[1] is x
    shorhs = np.array([-omega*omega*y[1], y[0]])
    return shorhs
#a=3.
#b=1.
#ab = np.array([a, b])
def test():
    amp = 1.0
    omega = 2.*pi/5.
    phase =0.
    tiny = 1.e-30
    params = np.array([amp, omega, phase])
    t=np.zeros(1000)
    y=np.zeros((len(t),2))
    h=1.e-2
    yn=np.zeros(2)
    yn[1]=1.0
    tn=0.0
    yscale =np.absolute(yn)+np.absolute(np.multiply(h,sho(tn,yn,omega)))+tiny
    epsilon=1.e-4
    for n in range(0,len(t)):
        yscale =np.absolute(yn)+np.absolute(h*sho(tn,yn,omega))+tiny
        #print(n,yscale)
        t[n]=tn
        y[n,:]=yn
        #tn,yn =rk4(tn,yn,h,sho,omega)
        #print(tn,yn,h,"hello")
        tn,yn,h=adaptiveRK4(tn,yn,h,sho,linearMaxFunc,omega,yscale,epsilon)
        #print(tn,yn,h)
    pyplot.figure()
    pyplot.xlabel("Time")
#    tn,yn,h=adaptiveRK4(tn,yn,h,sho,omega,yscale,epsilon)

#pyplot.ylabel("Relative error from analytic solution")
    #pyplot.title("RK4 solution to dx/dt = omega*cos(omega*t)")
    #pyplot.ylim([-1.,1.])
#    pyplot.xlim([0.,10.])
#    pyplot.plot(t,y[:,1])
#    pyplot.plot(t,np.cos(omega*t))
   
    pyplot.plot(t, y[:,1]-np.cos(omega*t))

#pyplot.plot(t,(y[:,1]-0.5*a*t*t-b*t)/(0.5*a*t*t+b*t))
    pyplot.show()

    #tested by comparison to Mathematica using "testrule"
    #x=np.array([1.27,1.86,.3,.2])
    #Rs=.3
    #print gammat(x,Rs)
    #print gammar(x,Rs)
    #print gammatheta(x,Rs)
    #print gammaphi(x,Rs)
    return 0
    
#sky is 4096 by 2048
skypixelheight = 2048
skypixelwidth = 4096

def main():
    #image plane's center is at Rplane<Router (radius of outer shell)
    h=1.e-3
    #h=1.e-4
    Router = 1000.
    Rplane = 700.
    Rs = 1.
    #imagewidth = 19.94175024251333s
    #imageheight = 10.73786551519949
    #inside horizon
    #pixelwidth =13
    #pixelheight =7
    pixelwidth = 101
    pixelheight = 51
    deltalamb = 1.e-1
    #epsilon = 1.e-6
    #yscale = [500.,500.,pi,2.*pi,-1.,1.,1.,1.]
    imagewidth = 20;
    imageheight = 10;
    tiny = 1.e-30
    epsilon=1.e-8
    pixelcoord = np.array([101,51])
    #coords = initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,imagewidth,imageheight,Rs)
    eccentricity = 0.2
    semilatusr = 10.0
    #HERE
    coords = initializeElliptical(eccentricity,semilatusr,Rs)
    print coords
    r=coords[1]
    lamb=0.
    color = 1
    n=0
    phi=coords[3]
#    while(False):
    #while(r<=Router):
    while(phi<2.*pi):
    #while(n<2):
        #lamb,coords,deltalamb =adaptiveRK4(lamb,coords,deltalamb,geodesic,Rs,yscale,epsilon)
        yscale =np.absolute(coords)+np.absolute(h*geodesic(lamb,coords,Rs))+tiny
        #lamb,coords =rk4(lamb,coords,deltalamb,geodesic,Rs)
        lamb,coords,h=adaptiveRK4(lamb,coords,h,geodesic,linearMaxFunc,Rs,yscale,epsilon)
        r=coords[1]
        phi=coords[3]
        if r<Rs:
            color = 0
            break
        #handle maximum number of integrations
        if(n%100==0):
            print(n,coords[0],r,coords[2],phi,h)
        n+=1
    #I am not sure if the following is correct
    print coords
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
        
    if (color==1):
        print("color=sky")
    else:
        print("color=blackhole")
    rmRs2 = coords[1]-Rs
    testnull = -rmRs2/coords[1]*coords[4]*coords[4]+coords[5]*coords[5]*coords[1]/rmRs2+coords[1]*coords[1]*(coords[6]*coords[6]+sin(coords[2])*sin(coords[2])*coords[7]*coords[7])
    print(testnull)
    print(testnull/max(np.absolute(coords[4:8])))
main()
#test()
