"""
Generate bar plots for the marginal probabilities estimated by a SMC sampler,
as in Schäfer and Chopin (2013).

We plot the min/max (across independent runs) of these probabilities, to assess
visually the Monte Carlo error. 

Note: first run concrete.py or boston.py to generate the results.

"""


import matplotlib.pyplot as plt

import numpy as np

marg_probs = []
for r in results:
    alg = r['output']
    pm = np.average(alg.X.theta, axis=0, weights=alg.W)
    marg_probs.append(pm)
marg_probs = np.array(marg_probs)

mp_max = marg_probs.max(axis=0)
mp_min = marg_probs.min(axis=0)

sorted_plot = False  # whether to sort the variables according to their marginal prob
if sorted_plot:
    idx = np.argsort(mp_max)
    mp_max = mp_max[idx]
    mp_min = mp_min[idx]

show_var_names = False
if show_var_names:
    pred_labels = list(cols.keys())
else:
    pred_labels = np.arange(1, npreds + 1)

# PLOTS
#######
plt.style.use('ggplot')
fig, ax = plt.subplots()

bar_max = ax.bar(pred_labels, mp_max, color='tab:red')
bar_min = ax.bar(pred_labels, mp_min, color='tab:blue')
ax.set_ylabel('marginal post probability')
ax.legend(handles=[bar_min, bar_max], labels=['min', 'max'])

plt.show()

