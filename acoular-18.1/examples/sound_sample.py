""""beamforming example for frame N_00001.wav"""


#import python as a framework

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import array, linspace,sin,cos,pi,zeros,long

#import acoular library

from os import path
from acoular import __file__ as bpath, WriteH5
from acoular import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
BeamformerTimeSqTraj,BeamformerCapon,BeamformerMusic,BeamformerDamas,BeamformerCMF,BeamformerClean,BeamformerFunctional,\
TimeCache, FiltOctave, BeamformerTime, TimePower, IntegratorSectorTime, \
PointSource, MovingPointSource, SineGenerator, WNoiseGenerator, Mixer, WriteWAV

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, \
psd, axis, xlim, ylim, title, tight_layout, text


freq = 48000
sfreq=freq/2 #sampling frequency
duration = 1
nsamples = freq*duration
datafile = 'cry_n0000001.wav'
micgeofile = path.join(path.split(bpath)[0],'xml','array_64_8mic.xml')
h5savefile = 'cry_n0000001.wav'
m = MicGeom(from_file=micgeofile)
n = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=1) #1pascal
p= PointSource(signal=n, mpos=m,  loc=(-0.1,-0.1,0.3))
p1 = Mixer(source=p)
wh5 = WriteH5(source=p, name=h5savefile)
wh5.save()

#definition the different source signal
r=52.5
nsamples = long(sfreq*0.3)
n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples)
s1 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq)
s2 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq, \
    phase=pi)

#define a circular array of 8 microphones
m = MicGeom('array_64_8mic.xml')
m.mpos_tot = array([(r*sin(2*pi*i+pi/8), r*cos(2*pi*i+pi/8), 0) \
    for i in linspace(0.0, 1.0, 8, False)]).T
t = MaskedTimeSamples(name=datafile)
f = PowerSpectra(time_data=t, window='Hanning', overlap='50%', block_size=128, \
    ind_low=1,ind_high=30)
g = RectGrid(x_min=0-.2, x_max=0.3, y_min=-0.13, y_max=0.3, z=0,
             increment=0.05)
bb = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04)
bc = BeamformerCapon(freq_data=f, grid=g, mpos=m, c=346.04, cached=False)
be = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, n=54)
bm = BeamformerMusic(freq_data=f, grid=g, mpos=m, c=346.04, n=6)
bd = BeamformerDamas(beamformer=bb, n_iter=100)
bo = BeamformerOrth(beamformer=be, eva_list=list(range(38,54)))
bs = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04)
bcmf = BeamformerCMF(freq_data=f, grid=g, mpos=m, c=346.04, \
    method='LassoLarsBIC')
bl = BeamformerClean(beamformer=bb, n_iter=100)
bf = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, \
    gamma=4)

# plot result maps for different beamformers in frequency domain

figure(1,(10,6))
i1 = 1
for b in (bb, bc, be, bm, bl, bo, bs, bd, bcmf, bf):
    subplot(3,4,i1)
    i1 += 1
    map = b.synthetic(freq,1)
    mx = L_p(map.max())
    imshow(L_p(map.T), origin='lower', vmin=mx-15,
           interpolation='nearest', extent=g.extend())
    colorbar()
    title(b.__class__.__name__)


