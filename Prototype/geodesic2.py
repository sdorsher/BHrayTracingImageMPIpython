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
    r = sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    theta = atan(y/x)
    phi = acos(z/r)

    #initial u perpendicular to plane.
    #magnitude of u is arbitrary-- affine parameter makes it rescalable
    #(ut)^2-(uy)^2-(ux)^2-(uz)^2=0 so ut = +-ux
    #for x decreasing as t increases, ut = -ux (inward)
    uy = 0.
    uz = 0.
    ux = 1.
    ur = sin(theta)*cos(phi)
    utheta= cos(theta)*cos(phi)
    uphi = -sin(theta)
    ut = r/(r-Rs)*sqrt(r*(r-Rs)*pow(sin(phi),2)*pow(sin(theta),2)+pow(cos(phi),2)*(r*(r-Rs)*pow(cos(theta),2)+pow(sin(theta),2)))
    initcoords =np.array([t, r, theta, phi, ut, ur, utheta, uphi])
    return initcoords
    
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
    return 0.5/pow(r,3)/(r-Rs)*(2.*r*pow(Rs,2)*pow(x[4],2)-pow(Rs,3)*pow(x[4],2)+pow(r,2)*Rs*(pow(x[5],2)-pow(x[4],2))+2.*pow(r,5)*pow(x[6],3)-4*pow(r,4)*Rs*pow(x[6],2)+2.*pow(r,3)*pow((r-Rs),2)*pow(sin(theta),2)*pow(x[7],2))
    
def RHStheta(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    return -2.*x[5]*x[6]/r+cos(theta)*sin(theta)*pow(x[7],2)
    
def RHSphi(x,Rs):
    t=x[0]
    r=x[1]
    theta=x[2]
    phi=x[3]
    return -2.*(x[5]+r*cos(theta)/sin(theta)*x[6])*x[7]/r

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


def main():
    #image plane's center is at Rplane<Router (radius of outer shell)
    Router = 1000.
    Rplane = 700.
    Rs = 1.
    #imagewidth = 19.94175024251333
    #imageheight = 10.73786551519949
    pixelwidth =13
    pixelheight =7


    pixelcoord = np.array([6,3])
    coords = initialize(pixelcoord,Rplane,pixelheight,pixelwidth,skypixelwidth,skypixelheight,Rs)
    print coords
    #do something about r crossing zero
    r=coords[1]
    deltalamb = 0.01
    lamb=0.
    color = 1.
    while(r<Router):
        lamb,coords =rk4(lamb,coords,deltalamb,geodesic,Rs)
        r=coords[1]
        if r<Rs:
            color = 0.
            break
        
    print coords


main()
