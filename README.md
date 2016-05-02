# LSUcompPhysFinalProject
LSU computational physics 2 final project

geodesic6.py produces output over the entire history of the geodesic for null or eliptical depending what is commented out.

geodesic11.py produces a skymap of what is seen by an observer with a telescope for null geodesics only. Currently it only shows where the black hole is. 

geodesic12.py also produces a skymap using flat row flat pixel, with the while loop separated into a function, with global variables for the adaptive rk4 constants. 

profile with 
python -m cProfile -s cumtime ../Prototype/geodesic12.py > prof12b.out
