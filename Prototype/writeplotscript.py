f=open("OutDirBigger/plotscript.gp","w")
pixelwidth = 1001
pixelheight= 1001
every = 100
f.write("splot [-1000:1000][-1000:1000][-1000:1000] \"output{xp}_{yp}.txt\" u (($3)*cos($5)*sin($4)):(($3)*sin($5)*sin($4)):(($3)*cos($4)) w l notitle\n".format(xp=1,yp=1))
for ypix in range(1,pixelheight,every):
    for xpix in range(1,pixelwidth,every):
        f.write("replot \"output{xp}_{yp}.txt\" u (($3)*cos($5)*sin($4)):(($3)*sin($5)*sin($4)):(($3)*cos($4)) w l notitle\n".format(xp=xpix,yp=ypix))
f.close()
