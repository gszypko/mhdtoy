#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

vec_interval = 5 #interval between B field vectors to display

input_file = open(sys.argv[1])
display_interval = float(sys.argv[2])
output_var = sys.argv[3] #must be one of rho, temp, press, energy, rad
vec_var = ""
if len(sys.argv) >= 5:
  vec_var = sys.argv[4]

#determine grid size
dim = input_file.readline().split(',')
xdim = int(dim[0])
ydim = int(dim[1])

#read in static magnetic field
bx = np.zeros([xdim,ydim], dtype=float)
by = np.zeros([xdim,ydim], dtype=float)
rows_x = input_file.readline().split(';')
rows_y = input_file.readline().split(';')
for i in range(xdim):
  bx[i] = np.asarray(rows_x[i].split(','))
  by[i] = np.asarray(rows_y[i].split(','))

var = []
vec_x = []
vec_y = []
t = []
output_number = 0
line = ""

while True:
  #read through to next time step (or file end)
  line = input_file.readline()
  while line and line[0:2] != "t=":
    line = input_file.readline()
  if not line:
    break
  time = float(line.split('=')[1])

  #read through to correct variable
  #vector quantities must be last in each time step
  line = input_file.readline()
  while line and line.rstrip() != output_var:
    if line[0:2] == "t=":
      sys.exit("Specified output variable not found in file")
    input_file.readline()
    line = input_file.readline()
  if not line:
    break

  if output_number == 0 or time/display_interval >= output_number:
    output_number += 1
    t.append(time)
    this_var = np.zeros([xdim,ydim], dtype=float)
    rows = input_file.readline().split(';')
    if len(rows) != xdim: break
    for i in range(xdim):
      this_var[i] = np.asarray(rows[i].split(','))
      if len(this_var[i]) != ydim: break
    var.append(this_var)

    if vec_var != "" and vec_var != "b":
      line = input_file.readline()
      while line and line.rstrip().split('_')[0] != vec_var:
        if line[0:2] == "t=":
          sys.exit("Specified output vector not found in file")
        input_file.readline()
        line = input_file.readline()
      if not line:
        break

      if output_number == 1 or time/display_interval >= (output_number-1):
        this_vec_x = np.zeros([xdim,ydim], dtype=float)
        rows = input_file.readline().split(';')
        if len(rows) != xdim: break
        for i in range(xdim):
          this_vec_x[i] = np.asarray(rows[i].split(','))
          if len(this_vec_x[i]) != ydim: break
        this_vec_y = np.zeros([xdim,ydim], dtype=float)
        input_file.readline()
        rows = input_file.readline().split(';')
        if len(rows) != xdim: break
        for i in range(xdim):
          this_vec_y[i] = np.asarray(rows[i].split(','))
          if len(this_vec_y[i]) != ydim: break
        vec_x.append(this_vec_x)
        vec_y.append(this_vec_y)
  
input_file.close()

if len(var) > len(vec_x):
  print("pop!")
  var.pop()

# fig = plt.figure()
fig, ax = plt.subplots()

frame = 0
# im = plt.imshow(np.transpose(var[frame]), animated=True, origin='lower', interpolation='bilinear')
# fig.colorbar(im)
# im = ax.imshow(np.transpose(var[frame]), animated=True, origin='lower', interpolation='nearest', vmin=0.0, vmax=(np.max(var[0])+1.0))
if output_var != "rad":
  im = ax.imshow(np.transpose(var[frame]), animated=True, origin='lower', \
    interpolation='nearest', norm=matplotlib.colors.LogNorm())
  ax.set(xlabel="x",ylabel="y",title=output_var+", t="+str(t[frame]))
  fig.colorbar(im)
else:
  im = ax.imshow(np.transpose(var[frame]), animated=True, origin='lower', \
    interpolation='nearest', norm=matplotlib.colors.SymLogNorm(linthresh=1e-5, base=10))
  ax.set(xlabel="x",ylabel="y",title=output_var+", t="+str(t[frame]))
  im.set_clim(vmin=1e-4)
  fig.colorbar(im)

contour_color_axes = fig.axes[-1]

X, Y = np.mgrid[0:xdim, 0:ydim]
# ax.streamplot(np.transpose(X),np.transpose(Y),np.transpose(bx),np.transpose(by),color='k',density=2.0/3.0)
if vec_var == "b": ax.quiver(X[::vec_interval, ::vec_interval], \
  Y[::vec_interval, ::vec_interval], \
    bx[::vec_interval, ::vec_interval], \
      by[::vec_interval, ::vec_interval], pivot='mid')
elif vec_var != "":
  this_vec_x = vec_x[frame].copy()
  this_vec_y = vec_y[frame].copy()
  norm = np.sqrt(this_vec_x**2 + this_vec_y**2)
  np.divide(this_vec_x, norm, out=this_vec_x, where=norm > 0)
  np.divide(this_vec_y, norm, out=this_vec_y, where=norm > 0)
  quiv = ax.quiver(X[::vec_interval, ::vec_interval], \
    Y[::vec_interval, ::vec_interval], \
      this_vec_x[::vec_interval, ::vec_interval], \
        this_vec_y[::vec_interval, ::vec_interval], \
          norm[::vec_interval, ::vec_interval], \
            cmap=plt.cm.binary, scale_units='xy', angles='xy', scale=0.3, \
              norm=matplotlib.colors.SymLogNorm(linthresh=1e4, base=10))
  fig.colorbar(quiv)
  vector_color_axes = fig.axes[-1]
  # ax.quiverkey(quiv, X=0.3, Y=1.1, U=10,
  #   label='Quiver key, length = 10', labelpos='E')


def updatefig(*args):
    global frame
    # global im
    global ax
    global quiv
    frame = (frame + 1)%len(var)
    im.set_data(np.transpose(var[frame]))
    im.autoscale()
    contour_color_axes.cla()
    fig.colorbar(im, cax=contour_color_axes)
    ax.set(xlabel="x",ylabel="y",title=output_var+", t="+str(t[frame]))
    if vec_var != "b" and vec_var != "":
      vector_color_axes.cla()
      fig.colorbar(quiv, cax=vector_color_axes)
      this_vec_x = vec_x[frame].copy()
      this_vec_y = vec_y[frame].copy()
      norm = np.sqrt(this_vec_x**2 + this_vec_y**2)
      np.divide(this_vec_x, norm, out=this_vec_x, where=norm > 0)
      np.divide(this_vec_y, norm, out=this_vec_y, where=norm > 0)
      quiv.set_UVC(this_vec_x[::vec_interval, ::vec_interval], \
          this_vec_y[::vec_interval, ::vec_interval], \
            norm[::vec_interval, ::vec_interval])
      quiv.autoscale()
    return im, ax

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=False)
plt.show()
