import numpy as np
from mpi4py import MPI

import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import smc_samplers as ss


# ============================================================
# 1. Define a simple state-space model
# ============================================================

class SimpleSSM(ssm.StateSpaceModel):

    default_params = {
        "a": 0.5,
        "sigma_x": 0.2,
        "sigma_y": 0.3,
    }

    def PX0(self):
        return dists.Normal(loc=0.0, scale=1.0)

    def PX(self, t, xp):
        return dists.Normal(
            loc=self.a * xp,
            scale=self.sigma_x
        )

    def PY(self, t, xp, x):
        return dists.Normal(
            loc=x,
            scale=self.sigma_y
        )


# ============================================================
# 2. Initialize MPI
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================
# 3. Generate synthetic observations
# ============================================================

T = 100

if rank == 0:

    rng = np.random.default_rng(42)

    a_true = 0.8
    sigma_x = 0.2
    sigma_y = 0.3

    x = np.zeros(T)
    y = np.zeros(T)

    x[0] = rng.normal(0.0, 1.0)
    y[0] = x[0] + rng.normal(0.0, sigma_y)

    for t in range(1, T):
        x[t] = a_true * x[t - 1] + rng.normal(0.0, sigma_x)
        y[t] = x[t] + rng.normal(0.0, sigma_y)

else:
    y = None

# All processes receive the same observations.
y = comm.bcast(y, root=0)


# ============================================================
# 4. Define the prior distribution
# ============================================================

prior = dists.StructDist({
    "a": dists.Uniform(a=0.0, b=1.0)
})


# ============================================================
# 5. Initialize SMC²
# ============================================================

Ntheta = 80
Nx = 50

fk_smc2 = ss.SMC2(
    ssm_cls=SimpleSSM,
    prior=prior,
    data=y,
    init_Nx=Nx,
    ar_to_increase_Nx=0.1,
    mpi=1,
    wastefree=False,
    len_chain=2
)


# ============================================================
# 6. Execute SMC²
# ============================================================

if rank == 0:

    print(f"Running SMC² with {size} MPI processes")

    alg = particles.SMC(
        fk=fk_smc2,
        N=Ntheta,
        verbose=True
    )

    alg.run()

    # Compute posterior mean of parameter a.
    posterior_mean = np.average(
        alg.X.theta["a"],
        weights=alg.W
    )

    print("\n========== RESULTS ==========")
    print("True parameter:", a_true)
    print("Estimated parameter:", posterior_mean)
    print("=============================")

    # Explicitly notify workers that the algorithm has finished.
    comm.bcast({"stop": True}, root=0)

else:

    # Worker processes execute requests from rank 0.
    while True:

        request = comm.bcast(None, root=0)

        if request.get("stop", False):
            break

        fk_smc2.run_pf_mpi(
            request["N"],
            request["Nx"],
            request["t"],
            request["move"]
        )