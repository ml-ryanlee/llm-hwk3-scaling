from __future__ import annotations

import json
import logging
import os
from typing import Optional
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# open json
with open("../data/isoflops_curves.json", "r") as f:
    data = json.load(f)

# best loss for budget, opt n for budget, opt d for budget 
best_loss = defaultdict(lambda: float('inf'))
best_n = defaultdict(int)
best_d = defaultdict(int)

for entry in data:
    num_params = entry["parameters"]
    budget = entry["compute_budget"]
    loss = entry["final_loss"]

    if loss < best_loss[budget]:
        best_loss[budget] = loss
        best_n[budget] = num_params # C = 6ND -> D = C / (6N)
        best_d[budget] = round(budget / (6*num_params)) 

# use curve fit to fit a polynomial function
def func(x,a,b):
    return a*x**b

x_flops = list(best_n.keys())
y_params = list(best_n.values())
y_tokens = list(best_d.values())

popt_n, _ = curve_fit(func,x_flops,y_params)
popt_d, _ = curve_fit(func,x_flops,y_tokens)

print("Coeff of Optimal Params line: ", popt_n)
print("Coeff of Optimal Tokens line: ", popt_d)
print("Optimal Num Params (B) for 10^23 flop budget ", round(func(1e23,popt_n[0],popt_n[1])/1e9))
print("Optimal Num Tokens (B) for 10^23 flop budget ", round(func(1e23,popt_d[0],popt_d[1])/1e9))

print("Optimal Num Params (B) for 10^24 flop budget ", round(func(1e24,popt_n[0],popt_n[1])/1e9))
print("Optimal Num Tokens (B) for 10^24 flop budget ", round(func(1e24,popt_d[0],popt_d[1])/1e9))

# create fitted curves with the coefficients found with curvefit
min_budget = min(x_flops)
x_proj = np.linspace(min_budget,1e25,100)
y_proj_n = [func(x,popt_n[0],popt_n[1]) for x in x_proj]
y_proj_d = [func(x, popt_d[0], popt_d[1]) for x in x_proj]

# figure 1 params vs. flops
fig1, ax1 = plt.subplots()
ax1.plot(x_flops, y_params, 'o')
ax1.plot(x_proj, y_proj_n, '-')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y/1e9:.0f}B'))

ax1.set_xlabel("Compute Budget (FLOPs)")
ax1.set_ylabel("Optimal Size (Parameters)")
ax1.set_title("IsoFLOP Curve Parameters vs. FLOPs")
ax1.grid(True)
fig1.savefig('parameters_vs_flops.png', dpi=300, bbox_inches='tight')

# figure 2 tokens vs. flops
fig2, ax2 = plt.subplots()
ax2.plot(x_flops, y_tokens, 'o')
ax2.plot(x_proj, y_proj_d, '-')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y/1e9:.0f}B'))

ax2.set_xlabel("Compute Budget (FLOPs)")
ax2.set_ylabel("Optimal Data (Tokens)")
ax2.set_title("IsoFLOP Curve Tokens vs. FLOPs")
ax2.grid(True)
fig2.savefig('tokens_vs_flops.png', dpi=300, bbox_inches='tight')