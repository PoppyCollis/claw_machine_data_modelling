import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load & normalize
df = pd.read_csv('data.csv')
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
)

# 2. Extract x and columns
x = df['x']
true_col   = 'truedata'
lower_col  = 'truelower'
upper_col  = 'trueupper'

# 3. Poster‐style fonts + square figure
plt.rcParams.update({
    'font.size': 80,
    'axes.titlesize': 0,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 30,
})
fig, ax = plt.subplots(figsize=(8, 8))   # square!

# 4. True data + error band (only this gets a legend entry)
ax.plot(x, df[true_col],
        color='grey', linewidth=4,
        label='True data')
ax.fill_between(x,
                df[lower_col],
                df[upper_col],
                color='lightgrey', alpha=0.5)

# 5. Other metrics (no legend entries)
ax.plot(x, df['diff'],              color='#2066a8', linewidth=3)
ax.plot(x, df['entropy_attention'], color='#8ec1da', linewidth=3)
ax.plot(x, df['entropy'],           color='#ae282c', linewidth=3)

# 6. Ticks & labels
ax.set_xticks([-100, -50, 0, 50, 100])
ax.set_yticks([1, 2, 3, 4])
ax.set_xlabel('Target x Location')
ax.set_ylabel('Confidence')

# 7. Only bottom & left spines; remove top/right borders
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
# Thicken the remaining spines
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.5)

ax.grid(False)

# 8. Legend & layout (only shows True data)
ax.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.show()
