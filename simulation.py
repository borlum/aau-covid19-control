import numpy as np
import pandas as pd

import models
import controls

SEIR = models.SEIR()
opt = controls.OptimalControl(SEIR, H=30)

F_RK4 = SEIR.get_rk4_single_step()
MPC = opt.get_MPC()

t_incubation = 5.1
t_infective = 3.3
R0 = 2.4
N = 100000

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

# Simulate
for k in range(N_days-1):
    U_sim[:,k] = MPC(X_sim[:,k], p0)
    X_sim[:,k+1] = F_RK4(X_sim[:,k], U_sim[:,k], p0).toarray().flatten()

# Results
sim_result = pd.DataFrame(
    np.vstack((X_sim, U_sim)).T, 
    columns=SEIR.x.keys() + [SEIR.u.name()]
)