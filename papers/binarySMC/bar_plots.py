"""
Generate bar plots for the marginal probabilities estimated by a SMC sampler,
as in Schäfer and Chopin (2013).

We plot the min/max (across independent runs) of these probabilities, to assess
visually the Monte Carlo error. 

Note: first run concrete.py or boston.py to generate the results, and save them
in a pickle file.

"""


import matplotlib.pyplot as plt
import numpy as np
import pickle

# load results
xp_name = 'boston'  # boston, concrete, or toy
with open(f'{xp_name}.pkl', 'rb') as f:
    dres = pickle.load(f)

marg_probs = dres['marg_probs']
pred_names = dres['pred_names']
npreds = len(pred_names)

# compute min/max
mp_max = marg_probs.max(axis=0)
mp_min = marg_probs.min(axis=0)

sorted_plot = False  # whether to sort the variables according to their marginal prob
if sorted_plot:
    idx = np.argsort(mp_max)
    mp_max = mp_max[idx]
    mp_min = mp_min[idx]

show_var_names = True
if show_var_names:
    pred_labels = pred_names
else:
    pred_labels = np.arange(1, npreds + 1)

# PLOTS
#######
plt.style.use('ggplot')
fig, ax = plt.subplots()

bar_max = ax.barh(pred_labels, mp_max, color='tab:red')
bar_min = ax.barh(pred_labels, mp_min, color='tab:blue')
for item in ax.get_yticklabels():
    item.set_fontsize(6) 
ax.set_ylabel('marginal post probability')
ax.legend(handles=[bar_min, bar_max], labels=['min', 'max'])

plt.show()
plt.savefig(f'{xp_name}_bar_plot_marg_probs.pdf')
