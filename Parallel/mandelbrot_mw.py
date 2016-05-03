####################################################
#  Calculate Mandelbrot set and save it as a bmp image
#
#  Parallel version uses master-worker approach,
#  with master node at rank zero; the work units
#  distributed to jobs_completed_by_worker consist of a single
#  image next_row_to_assign. It needs at least two MPI nodes to work.
#
####################################################
 
import mpi
import bmp
 
# maximal number of iterations to compute a pixel
MAX_ITER = 256
 
# pixel computation function
def pixel(c):
    z = 0
    for i in range(MAX_ITER):
        z = z*z+c
        if abs(z) >= 2.0:
            return i
    return MAX_ITER
 
# image dimensions
nx = 1024
ny = 1024
 
jobs_completed_by_worker = []
for i in range(mpi.size):
    jobs_completed_by_worker.append(0)
 
if mpi.rank == 0:
    # "master" node:
 
    workers_running = 0
    next_row_to_assign  = 0
 
    # initialize list of image rows
    image = []
    for i in range(ny):
        image.append(-1)
 
    # get all workers started on tasks
    for n in range(1, mpi.size):
        mpi.send(next_row_to_assign, n)
        next_row_to_assign += 1
        workers_running += 1
 
    # master's main loop:
    while workers_running > 0:
        # receive computed result from any worker
        result, status = mpi.recv(mpi.ANY_SOURCE)
        worker_id = status.source
        row_completed, row_data = result
 
        jobs_completed_by_worker[worker_id] += 1
 
        # incorporate newly computed next_row_to_assign into image data
        image[row_completed] = row_data
 
        if next_row_to_assign < ny:
            # send new work unit to the (now) idle worker
            mpi.send(next_row_to_assign, worker_id) 
            next_row_to_assign += 1
        else:
            # use -1 as the row number to signal all done
            mpi.send(-1, worker_id)
            workers_running -= 1
 
    # convert data to color image and save it in a file
    bmp.write_image('image.bmp', nx, ny, image, MAX_ITER)
    for w in range(1,mpi.size):
        print jobs_completed_by_worker[w],"tasks completed by worker",w
 
else:
    # "worker" node:
    while 1:
        # receive work unit info
        row, status = mpi.recv(mpi.ANY_SOURCE)
        # check if we're still needed
        if row == -1:
            break
        # compute row of image
        rdata = []
 
        # Magic here... 4.0j is a complex number
        c = 4.0j*row/ny-2.0j
        for x in range(nx):
            rdata += [pixel(c+4.0*x/nx-2.0)]
        # send the result to master
        mpi.send([row, rdata], 0)