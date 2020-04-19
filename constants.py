import numpy as np

rho = 1.225  # kg/m^3
S = 2  # m^2
m = 2.5  # kg
g = 9.81 # m/s/s

# Rademacher (2009)
taus = 1.0
taue = 0.8

# Brown (1993)
Cltrim = 0.5  # [0.4 1.0]
Cdtrim = 0.09
delCl = 0.467
delCd = 0.19

Q = np.diag([1, 1, 1, 10, 10, 10])
Qf = np.diag([10, 1, 1, 10, 10, 10])
R = np.diag([1, 5])

n_x = 6
n_u = 2