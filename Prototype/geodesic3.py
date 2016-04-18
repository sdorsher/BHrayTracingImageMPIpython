import numpy as np
from math import pi,sin,cos,sqrt,atan,acos
from matplotlib import pyplot

# t is lamb, affine parameter
# y is u, four velocity, or x, four position

def initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,Rs):
    #set origin of pixel plane at x axis
    imagewidth = float(pixelwidth)*Rplane*pi*2./float(skypixelwidth)
    imageheight = float(pixelheight)*Rplane*pi/float(skypixelheight)
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

def adaptiveRK4(t,y,h,func,arg,yscale,epsilon):
    a=np.array([0, .2, .3, .6, 1., 7./8.])
    b=np.array([[0.,0.,0.,0.,0.],[.2,0.,0.,0.,0.],[3./40.,9./40.,0.,0.,0.],[3./10., -9./10., 6./5., 0., 0.],[-11./54., 2.5, -70./27., 35./27., 0.], [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
    c = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
    cstar = np.array([2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 0.25])
    safetyfac = 0.9
    pgrow = 0.20
    pshrink =0.25
    hnew=h/2.
    lena=len(a)
    leny=len(y)
    while True:
        # j and i are reversed from Numerical Recipes book (page 711)
        #loop over y indices
        k=np.zeros((leny,lena))
        tprimearg=t
        yprime = y
        yprimestar = y
        for j in range(0,len(a)): #for all terms summed in method
            tprimearg = t+a[j]*h
            #update variables for next k terms calculation
            yprimearg = y
            for n in range(0,leny): #over all variables in vector
                for i in range(0,j): #over all k indices
                    #update variables for next k terms calculation
                    yprimearg[n]+=b[j,i]*k[n,i]
            k[:,j]=h*func(tprimearg,yprimearg,arg)
        yprime = y+np.sum(np.multiply(c,k),axis=1)
        yprimestar =y+ np.sum(np.multiply(cstar,k),axis=1)
        delta1 = yprime - yprimestar
        temp1 = yscale/delta1
        errratio = np.absolute(np.multiply(epsilon,yscale/delta1))
        if min(errratio)>1:
            #print("shrink")
            hnew = safetyfac*h*min(np.power(errratio,pshrink))
            break
        else:
            hnew = safetyfac*h*min(np.power(errratio,pgrow))
            #problem has something to do with break conditions
            #print("breaking")
        h = hnew
        break
    #tprime = t+h
    tprime = t+hnew
    return tprime,yprimestar,hnew
    #return tprime,yprimestar,h

    
def rk4(t,y,h,func,arg):
    k1=h*func(t,y,arg) #no t on right hand side of these equations
    k2 = h*func(t+0.5*h,y+0.5*k1,arg)
    k3 = h*func(t+0.5*h,y+0.5*k2,arg)
    k4 = h*func(t+h,y+k3,arg)
    t=t+h
    return t,y+k1/6.+k2/3.+k3/3.+k4/6.

def geodesic(lamb,x,Rs):
    #returns a vector of the four acceleration
    dut = RHSt(x,Rs)
    dur = RHSr(x,Rs)
    dutheta = RHStheta(x,Rs)
    duphi = RHSphi(x,Rs)
    return np.array([x[4], x[5], x[6], x[7], dut, dur, dutheta, duphi])

def RHSt(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    return Rs*x[4]*x[5]/r/(Rs-r)
    
def RHSr(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    rmRs=r-Rs 
    cut = -Rs*rmRs/2./r/r/r
    cur = Rs/r/2./rmRs
    cutheta = rmRs
    cuphi = rmRs*sin(theta)*sin(theta)
    return cut*x[4]*x[4]+cur*x[5]*x[5]+cutheta*x[6]*x[6]+cuphi*x[7]*x[7]
    #    return 0.5/r/r/r/rmRs*(2.*r*Rs*Rs*x[4]*x[4]-Rs*Rs*Rs*x[4]*x[4]+r*r*Rs*(x[5]*x[5]-x[4]*x[4])+2.*r*r*r*r*r*x[6]*x[6]*x[6]-4.*r*r*r*r*Rs*x[6]*x[6]+2.*r*r*r*x[6]*x[6]+2.*r*r*r*rmRs*rmRs*sin(theta)*sin(theta)*x[7]*x[7])
    
def RHStheta(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    return -2.*x[5]*x[6]/r+cos(theta)*sin(theta)*x[7]*x[7]
    
def RHSphi(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    return -2.*(x[5]+r*cos(theta)/sin(theta)*x[6])*x[7]/r

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
    shorhs = np.array([-omega*y[1], omega*y[0]])
    return shorhs
#a=3.
#b=1.
#ab = np.array([a, b])
def test():
    amp = 1.0
    omega = 2.*pi/50.
    phase =0.
    params = np.array([amp, omega, phase])
    t=np.zeros(100000)
    y=np.zeros((len(t),2))
    h=1.e-2
    yn=np.zeros(2)
    yn[1]=1.0
    tn=0.0
    yscale =np.array([1.,1.])
    epsilon=1.e-3
    for n in range(0,len(t)):
        t[n]=tn
        y[n,:]=yn
        tn,yn =rk4(tn,yn,h,sho,omega)
#        tn,yn,h=adaptiveRK4(tn,yn,h,sho,omega,yscale,epsilon)
    print(yn)
    pyplot.figure()
    pyplot.xlabel("Time Step")
    #pyplot.ylabel("Relative error from analytic solution")
    #pyplot.title("RK4 solution to dx/dt = omega*cos(omega*t)")
    #pyplot.ylim([-1.,1.])
    
    #pyplot.plot(t,y[:,1])

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
    Router = 1000.
    Rplane = 700.
    Rs = 1.
    #imagewidth = 19.94175024251333
    #imageheight = 10.73786551519949
    #inside horizon
    #pixelwidth =13
    #pixelheight =7
    pixelwidth = 201
    pixelheight = 101
    deltalamb = 1.e-5
    #epsilon = 1.e-6
    #yscale = [500.,500.,pi,2.*pi,-1.,1.,1.,1.]

    pixelcoord = np.array([6,3])
    coords = initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,Rs)
    print coords
    r=coords[1]
    deltalamb = 1.e-3
    lamb=0.
    color = 1
    n=0
#    while(False):
    while(r<=Router):
        #lamb,coords,deltalamb =adaptiveRK4(lamb,coords,deltalamb,geodesic,Rs,yscale,epsilon)
        lamb,coords =rk4(lamb,coords,deltalamb,geodesic,Rs)
        r=coords[1]
        if r<Rs:
            color = 0
            break
        #handle maximum number of integrations
        if(n%10000==0):
            print(n,r,coords[2],coords[3])
        n+=1
    print(coords)
    #I am not sure if the following is correct
    if(coords[2]<0.):
        temp = (-coords[2])%(2.*pi)
        coords[2]=pi-temp
    else:
        coords[2]%=pi

    if(coords[3]<0.):
        temp=(-coords[3])%(2.*pi)
        coords[3]=2.*pi-temp
    else:
        coords[3]%=(2.*pi)
    print coords

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
