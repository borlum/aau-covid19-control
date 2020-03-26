import numpy as np
import pandas as pd

import time

import models
import controls

SEIR = models.SEIR()
opt = controls.OptimalControl(SEIR, H=30)

# CoVID19 properties
t_incubation = 5.1
t_infective = 3.3
R0 = 2.4

# Population (DK)
N = 1e5
max_infected = 2000 # % of N

opt.X_ub[2] = max_infected / N
opt.U_lb[0] = 0
opt.U_ub[0] = 0.8

# 

# Initial number of infected and recovered individuals
e0 = 1/N
i0 = 0
r0 = 0
s0 = 1 - e0 - i0 - r0
x0 = [s0, e0, i0, r0]

p0 = [1/t_incubation, R0/t_infective, 1/t_infective]

N_days = 365
X_sim = np.zeros((SEIR.x.size, N_days))
U_sim = np.zeros((1, N_days))
X_sim[:,0] = x0

# Get integrator
F_RK4 = SEIR.get_rk4_single_step()
# Get MPC
MPC = opt.get_MPC()

# Simulate
t0 = time.time()
init = 0
for k in range(N_days-1):
    U_sim[:,k], init = MPC(X_sim[:,k], p0, init)

    # Stochastic disturbance
    #p_noise = 0.1 * (p0 * np.random.randn(len(p0))) + p0
    # Also, discrepancy between desired u and applied u (population compliance)
    #U_sim[:,k] = 0.85 * U_sim[:,k]
    
    X_sim[:,k+1] = F_RK4(X_sim[:,k], U_sim[:,k], p0).toarray().flatten()

t1 = time.time()

print(t1 - t0)

# Results
sim_result = pd.DataFrame(
    np.vstack((X_sim, U_sim)).T, 
    columns=SEIR.x.keys() + [SEIR.u.name()]
)