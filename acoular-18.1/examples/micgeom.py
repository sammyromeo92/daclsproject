#from .version import __author__, __date__, __version__

import os

# workaround for problems with pyqt5 support, may be removed in the future
try:
    import pyface.qt
except:
    os.environ['QT_API'] = 'pyqt'

# make sure that no OMP multithreading is used if OMP_NUM_THREADS is not defined
os.environ.setdefault('OMP_NUM_THREADS','1')

#from .fileimport import time_data_import, csv_import, td_import, \
#bk_mat_import, datx_import
try:
    from .nidaqimport import nidaq_import
except:
    pass

object_name.configure_traits()
m = MicGeom(from_file='UCA8.xml')



import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import path
import acoular
from acoular import L_p, Calib, MicGeom, TimeSamples, \
RectGrid, BeamformerBase, EigSpectra, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional

object_name.configure_traits()
m = MicGeom(from_file='UCA8.xml')

m = MicGeom(from_file='UCA8.xml')
g = RectGrid(x_min=-0.8, x_max=-0.2, y_min=-0.1, y_max=0.3, z=0.8, increment=0.01)
t1 = TimeSamples(name='cry_n0000001.wav')
f1 = EigSpectra(time_data=t1, block_size=256, window="Hanning", overlap='75%')
e1 = BeamformerBase(freq_data=f1, grid=g, mpos=m, r_diag=False)
fr = 4000
L1 = L_p(e1.synthetic(fr, 0))

object_name.configure_traits()
m = MicGeom(from_file='UCA8.xml')
