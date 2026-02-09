#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_async_allocator=true"
import jax
jax.print_environment_info()
print("Devices:", jax.devices())


from pathlib import Path

root_path = Path.cwd().parent.absolute()
import sys

sys.path.append(str(root_path))
import os


# In[2]:


from flax import nnx
import jax.numpy as jnp


# In[3]:


from geometry.G_matrix import G_matrix
from functionals.functions import create_potentials
from functionals.linear_funcitonal_class import LinearPotential
from functionals.internal_functional_class import InternalPotential
from functionals.interaction_functional_class import InteractionPotential
from functionals.functional import Potential
from flows.anderson_acceleration import anderson_method
from flows.gradient_flow import run_gradient_flow
from parametric_model.parametric_model import ParametricModel


# # Define parametric model 

# In[4]:


# Define the parametric model
dim = 30
parametric_model = ParametricModel(
    parametric_map="node",  # "node" "resnet"
    rhs_model="mlp",  # "mlp" or "resnet"
    architecture=[dim, 2, 64],  # [input_dim, num_layers, hidden_width]
    activation_fn="tanh",  # "tanh", "relu", "SinTu", "identity", "sigmoid", "gelu", "swish"
    time_dependent=True,  # True or False
    solver="euler",  # "euler" or "heun"
    dt0=0.01, 
    ref_density="gaussian",
    scale_factor=1e0,
)
_, init_params  = nnx.split(parametric_model)

key = jax.random.PRNGKey(1234)
rngs = nnx.Rngs(key)

G_mat_flow = G_matrix(parametric_model)
solver = "minres"  # minres or cg


# # Define the $\operatorname{KL}$ minimization problem by choosing the target functional

# In[5]:


# case = 'st' | 'double banana' | 'gaussian'
case = 'st'


# ## Gaussian setup

# In[6]:


from functionals.functions import get_gaussian_potential
if case == 'gaussian':
    mean = jnp.full((dim,), 2.0)
    sigma_inv = jnp.diag(jnp.array([1000., 10.]))
    potential_fn = get_gaussian_potential(mean, sigma_inv)

    # Anderson parameters
    h = 1e-3 # Time step size
    m = 6  # Number of previous iterates to consider
    beta = 1.25  # Damping parameter
    reg = 1e-5
    max_iterations = 300
    tolerance = 1e-4
    n_samples = 3_000  # Monte Carlo sample size


# ## Double-banana setup

# In[7]:


if case == 'double banana':
    from jax.scipy.special import logsumexp
    from functools import partial

    def potential_fn_double_banana(x, shift):
        x -= shift[..., :]
        x1 = x[..., 0]
        
        log_density = 2.*(jnp.linalg.norm(x, axis=-1) - 3.) ** 2
        log_density -= logsumexp(jnp.stack([-2.*(x1 - 3.)**2, -2.*(x1 + 3.)**2], axis=-1), axis=-1)
        return log_density
        
    potential_fn = partial(potential_fn_double_banana, shift=jnp.array([0., 10.]))
    # Anderson parameters
    h = 5e-2  # Time step size
    m = 8  # Number of previous iterates to consider
    beta = 1.8 # Damping parameter
    reg = 1e-5 # l2 regularization of mixing parameters Gamma
    max_iterations = 500
    tolerance = 1e-4
    n_samples = 3000  # Monte Carlo sample size


# ## Styblinsky-Tang potential

# In[8]:


from functionals.functions import styblinski_tang_potential_fn
from functools import partial
if case == 'st':
    potential_fn = partial(styblinski_tang_potential_fn, d=dim)
    h = 1e-3  # Time step size
    n_samples = 10_000  # Monte Carlo sample size
    tolerance = 1e-4
    max_iterations = 3000
    # Anderson parameters
    m = 8  # Number of previous iterates to consider
    beta = 1.8 # Damping parameter
    reg = 1e-3 # l2 regularization of mixing parameters Gamma


# Define the $\operatorname{KL}$ functional (i.e. linear + entropy with equal coef.)

# In[9]:


linear_potential = LinearPotential(potential_fn=potential_fn, coeff=1.)
internal_potential = InternalPotential(
    functional="entropy", coeff=1.0, method="exact", prob_dim=dim
)

potential = Potential(
    linear=linear_potential, internal=internal_potential, interaction=None
)


# In[10]:


key, subkey = jax.random.split(key)
# <<test dataset>>
z_samples = jax.random.normal(
    subkey,
    (300, dim),
)


# # Gradient flow (Picard)
# Generate reference sample with Picard method

# In[11]:


init_params_gd = jax.tree.map(lambda _x: _x, init_params)
graphdep, _ = nnx.split(parametric_model)
pm_gd = nnx.merge(graphdep, init_params_gd)
history_gd = run_gradient_flow(
    pm_gd,
    z_samples,
    G_mat_flow,
    potential,
    N_samples=n_samples,
    max_iterations=max_iterations,
    h=h,
    # progress_every=max_iterations//4,
    progress_every=10,
    solver=solver,
)


# In[ ]:


final_model = history_gd.pop('final_parametric_model')
pot = history_gd.pop('potential')


# In[ ]:


import pickle
import time

timestamp = time.strftime("%d.%m.%Y_%H:%M.%S")

with open(f'./{timestamp}_gf_{case}.pkl', 'wb') as ofile:
    pickle.dump(history_gd, ofile)


# # Anderson acceleration

# In[ ]:


graphdep, _ = nnx.split(parametric_model)
init_params_am = jax.tree.map(lambda _x: _x, init_params)
pm_anderson = nnx.merge(graphdep, init_params_am)
cp_anderson, history_anderson = anderson_method(pm_anderson, 
                                       n_samples, 
                                       z_samples, 
                                       G_mat_flow, 
                                       potential,
                                       solver=solver,
                                       initial_params=init_params,
                                       n_iterations=max_iterations, 
                                       step_size=h, 
                                       memory_size=m, 
                                       relaxation=beta, 
                                       regularization_factor_gamma=reg,
                                       regularization_method_gamma='l2',
                                       ensure_descent=True)


# In[ ]:



with open(f'./{timestamp}_am_{case}.pkl', 'wb') as ofile:
    pickle.dump(history_anderson, ofile)


# # Plot results

# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(20,10), sharex=True)
ax = axs[0]
e_gd = history_gd["energy_history"]
e_gd = jnp.array(e_gd)
e_am = history_anderson["energies"]
e_am = jnp.array(e_am)
# e_am = e_gd
e_min = jnp.min(jnp.concatenate((e_gd, e_am))) - 1e-10
e_gd -= e_min  # so that we can safely plot in log scale
e_am -= e_min
ax.plot(e_gd, label='GD')
ax.plot(e_am, label='Anderson')
ax.set_yscale("log")
# ax.set_ylim(1e-2, None)
ax.grid()
ax.set_title(r'$\mathcal{E} - \mathcal{E}_{\text{min}}$')

ax = axs[1]

ax.plot(history_gd['riemann_grad_norm_history'], label='GD')
ax.plot(history_anderson['riemann_grad_history'], label='Anderson')
ax.set_yscale('log')
ax.grid()
ax.set_title(r"$\| \nabla_\theta F\|_{G(\theta)}$")

# ax = axs[2]
# ax.plot(history['euclid_grad_norm_history'], label='GD')
# ax.set_yscale('log')
# ax.grid()
# ax.set_title(r"$\| \nabla_\theta F\|_2$")

axs[-1].legend()
# ax.set_xlim((0, 30))
fig.tight_layout()
# fig.savefig('GD_vs_AM.pdf')


# In[ ]:



fig, ax = plt.subplots()


key, subkey = jax.random.split(key)
z_samples = jax.random.normal(subkey, shape=(200, dim))
gd, _ = nnx.split(parametric_model)
model_am = nnx.merge(gd, cp_anderson)
# sample_am = model_am(z_samples)
sample_am = jnp.zeros((1, dim))
ax.scatter(*sample_am[:, :2].T, label='AM')


sample_gd = final_model(z_samples)
ax.scatter(*sample_gd[:, :2].T, label='GD')

joint_sample = jnp.concat((sample_am, sample_gd), axis=0)
l = jnp.min(joint_sample, axis=0)
r = jnp.max(joint_sample, axis=0)

bds = jnp.stack((l, r)).T

potential.linear.plot_function(ax=ax, fig=fig, x_bds=bds[0], y_bds=bds[1])
ax.legend()

fig.savefig(f"{case}_scatter.pdf")


# In[ ]:


# import pickle

# with open('./09.12.2025_15:34.41_gf_gaussian.pkl', 'rb') as ifile:
    # history_gd = pickle.load(ifile)
    


# In[ ]:


# with open('./09.12.2025_15:34.41_am_gaussian.pkl', 'rb') as ifile:
    # history_anderson = pickle.load(ifile)


# In[ ]:




