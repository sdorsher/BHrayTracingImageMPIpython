f=open("OutDir/plotscript.gp","w")
pixelwidth = 101
pixelheight= 51
f.write("splot [-1000:1000][-1000:1000][-1000:1000] \"output{xp}_{yp}.txt\" u (($3)*cos($5)*sin($4)):(($3)*sin($5)*sin($4)):(($3)*cos($5)) w l notitle\n".format(xp=1,yp=1))
for ypix in range(1,pixelheight,int(pixelheight/10)):
    for xpix in range(1,pixelwidth,int(pixelwidth/10)):
        f.write("replot \"output{xp}_{yp}.txt\" u (($3)*cos($5)*sin($4)):(($3)*sin($5)*sin($4)):(($3)*cos($5)) w l notitle\n".format(xp=xpix,yp=ypix))
f.close()
