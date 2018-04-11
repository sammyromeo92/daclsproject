'''
This file describes the location of microphones on
a circular array of 8 microphones
'''

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

#n_mics = 8        # number of microphones
#diameter = 105    #  mm


# posizione angolare dei microfoni in frazioni di 2 \ pi
angles = np.array([0., 2., 4., 1., 5., 3.,4.,5.]) * 1./8.

# Array of microphones location sin 3D
R_compactsix_circular_1 = np.array([
  diameter*np.cos(2.*np.pi*angles),
  diameter*np.sin(2.*np.pi*angles),
  np.zeros(n_mics),])


from os import path
from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource, Mixer, WriteH5

sfreq = 48000
duration = 1
nsamples = duration*sfreq
micgeofile = path.join(path.split(bpath)[0],'xml','array_64_8mic.xml')
h5savefile = 'sammy.csv'

m = MicGeom(from_file=micgeofile)
n = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=1) #1pascal
p= PointSource(signal=n, mpos=m,  loc=(-0.1,-0.1,0.3))
p1 = Mixer(source=p)
wh5 = WriteH5(source=p, name=h5savefile)
wh5.save()

micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64_8mic.xml')
datafile = 'sammy.csv'

mg = acoular.MicGeom( from_file=micgeofile )
ts = acoular.TimeSamples( name='sammy.csv')
ps = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, overlap= '50%')
bb = acoular.BeamformerBase( freq_data=ps, grid=rg, mpos=mg )
pm = bb.synthetic( 8000, 3 )
Lm = acoular.L_p( pm )
imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), interpolation='bicubic')
colorbar()
figure(2)
plot(mg.mpos[0],mg.mpos[1],'o')
axis('equal')
show()

