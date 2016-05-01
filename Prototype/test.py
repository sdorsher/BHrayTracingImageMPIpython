import PIL,png
from scipy import misc
from matplotlib import pyplot,image
import numpy as np

f=open("test.png","w")

telepixels = np.zeros((101,101,3),dtype=np.uint8)
telepixels2 = np.zeros((101*101*3),dtype=np.uint8)

for xpix in range(48,52):
    for ypix in range(8,93):
        start = (xpix+ypix*101)*3
        telepixels2[start:start+3]=(255,0,0)
        telepixels[xpix,ypix]=np.array([255,255,255],dtype=np.uint8)

#png.from_array(telepixels,'L').save("test.png")
tele=png.Writer(width=101,height=101,greyscale=False,alpha=False)
tele.write_array(f,telepixels2)
f.close()
