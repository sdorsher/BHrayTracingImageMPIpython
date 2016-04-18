import numpy as np
from math import pi,sin,cos,sqrt,atan,acos
from matplotlib import pyplot

# t is lamb, affine parameter
# y is u, four velocity, or x, four position

def initialize(pixelcoord,Router,pixelheight,pixelwidth,skypixelwidth,skypixelheight):
    #set origin of pixel plane at x axis
    imagewidth = float(pixelwidth)*Router*pi*2./float(skypixelwidth)
    imageheight = float(pixelheight)*Router*pi/float(skypixelheight)
    t = 0.
    x = Router
    y = (pixelcoord[0]-pixelwidth/2.)*imagewidth/float(pixelwidth)
    z = (pixelcoord[1]-pixelheight/2.)*imageheight/float(pixelheight)
    r = sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    theta = atan(y/x)
    phi = acos(z/r)

    #initial u perpendicular to plane.
    #magnitude of u is arbitrary-- affine parameter makes it rescalable
    #(ut)^2-(uy)^2-(ux)^2-(uz)^2=0 so ut = +-ux
    #for x decreasing as t increases, ut = -ux (inward)
    uy =0.
    uz =0.
    ux =1.
    ut =-1.

    #how do you convert four velocities to polar coordinates?
    
def rk4(t,y,h,func):
    k1=h*func(t,y,z) #no t on right hand side of these equations
    k2 = h*func(t+0.5*h,y+0.5*k1,arg)
    k3 = h*func(t+0.5*h,y+0.5*k2,arg)
    k4 = h*func(t+h,y+k3,arg)
    return y+k1/6.+k2/3.+k3/3.+k4/6.

def geodesic(lamb,xu,Rs):
    x=xu[0:3]
    u=xu[4:7]
    #returns a vector of the four acceleration
    Gt = gammat(x,Rs)
    Gr = gammar(x,Rs)
    Gtheta = gammatheta(x,Rs)
    Gphi = gammaphi(x,Rs)
    dut = u.dot(Gt).dot(u)
    dur = u.dot(Gr).dot(u)
    dutheta= u.dot(Gtheta).dot(u)
    duphi = u.dot(Gphi).dot(u)
    return np.array([u[0], u[1], u[2], u[3], dut, dur, dutheta, duphi])


def gammat(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    f=Rs/(2.*pow(r,2.)-2.*r*Rs)
    Gt=np.array([[0,f,0,0],[f,0,0,0],[0,0,0,0],[0,0,0,0]])
    return Gt

def gammar(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    f = (r-Rs)*Rs/2./pow(r,3.)
    g = -Rs/2./r/(r-Rs)
    h = r-Rs
    k = h*sin(theta)*sin(theta)
    Gr = np.array([[f,0,0,0],[0,g,0,0],[0,0,h,0],[0,0,0,k]])
    return Gr
    
def gammatheta(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    f=1./r
    g=-cos(theta)*sin(theta)
    Gtheta=np.array([[0,0,0,0],[0,0,f,0],[0,f,0,0],[0,0,0,g]])
    return Gtheta

def gammaphi(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    f=1./r
    g=cos(theta)/sin(theta)
    Gphi=np.array([[0,0,0,0],[0,0,0,f],[0,0,0,g],[0,f,g,0]])
    return Gphi
    

#parabola test
#def parabola(t,y,ab):
#    ab[0] = a
#    ab[1] =b
#    return a*t+b
#a=3.
#b=1.
#ab = np.array([a, b])
#t=np.array(np.arange(0,100,0.01))
#y=np.array(np.zeros(len(t)))
#for n in range(0,len(t)-1):
#    y[n+1]=rk4(t[n],y[n],t[n+1]-t[n],parabola,ab)
#pyplot.figure()
#pyplot.xlabel("Time Step")
#pyplot.ylabel("Relative error from analytic solution")
#pyplot.title("RK4 solution to dx/dt = at+b")

#pyplot.plot(t/0.01,(y-0.5*a*pow(t,2.)-b*t)/(0.5*a*pow(t,2.)-b*t))
#pyplot.show()

#tested by comparison to Mathematica using "testrule"
#x=np.array([1.27,1.86,.3,.2])
#Rs=.3
#print gammat(x,Rs)
#print gammar(x,Rs)
#print gammatheta(x,Rs)
#print gammaphi(x,Rs)

#sky is 4096 by 2048
skypixelheight = 2048
skypixelwidth = 4096

#image plane's center is at Router (radius of outer shell)
Router = 1000.
Rs = 1.
#imagewidth = 19.94175024251333
#imageheight = 10.73786551519949
pixelwidth =13
pixelheight =7


pixelcoord = np.array([6,3])
initialcoords = initialize(pixelcoord,imageheight,Router,pixelheight,pixelwidth,skypixelwidth,skypixelheight)
