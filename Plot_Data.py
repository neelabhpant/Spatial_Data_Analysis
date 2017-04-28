from __future__ import division
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.stats as stats
import pylab as pl
import sqlite3
import matplotlib.pyplot as plt


conn = sqlite3.connect('kml_data.db');
c = conn.cursor();
statement = 'SELECT ZCURVE FROM ZCURVE limit 935'
c.execute(statement)
data = c.fetchall()
dec_z = [int(str(i[0]), 2) for i in data]
norm_dec_z = [decimal / max(dec_z) for decimal in dec_z]
fit = stats.norm.pdf(norm_dec_z, np.mean(norm_dec_z), np.std(norm_dec_z))
pl.plot(norm_dec_z,fit,'-o')
plt.hist(norm_dec_z, normed=True)
plt.xlabel("Z-Value")
plt.ylabel("Frequency")
plt.show()


conn2 = sqlite3.connect('kml_data.db');
c2 = conn2.cursor();
statement2 = 'SELECT HILBERT FROM HILBERT limit 1000'
c2.execute(statement2)
data_h = c2.fetchall()
data_hilbert = [int(data[0]) for data in data_h]
fit2 = stats.norm.pdf(data_hilbert, np.mean(data_hilbert), np.std(data_hilbert))
pl.plot(data_hilbert,fit2,'-o')
pl.hist(data_hilbert, normed=True)
pl.xlabel("Hilbert Value")
pl.ylabel("Frequency")
pl.show()