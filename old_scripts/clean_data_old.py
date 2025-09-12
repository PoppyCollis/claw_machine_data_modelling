import numpy as np
import pandas as pd

# Load
data_path = 'data/all_participants_action_selection_data.csv'
cols = ['participant_id', 'trial_number', 'first_option', 'second_option', 'chosen_type', 'action_selection_conf']
df = pd.read_csv(data_path, usecols=cols)

# Create a complete MultiIndex of (participant_id, trial_number)
pids = df['participant_id'].unique()
trials = np.arange(1, 109)       # 1 through 108
full_idx = pd.MultiIndex.from_product([pids, trials],
                                      names=['participant_id', 'trial_number'])

# Re-index, so missing rows become NaN
df = df.set_index(['participant_id', 'trial_number']).reindex(full_idx).reset_index()

# (Optional) report which trials were missing per participant
missing = (
    df[df['first_option'].isna()]
      .groupby('participant_id')['trial_number']
      .apply(list)
)
print("Missing trials per participant:\n", missing)

# Map your strings to ints (will leave NaN alone)
dist_map = {
    'narrow_low': 0,
    'wide_low':   1,
    'narrow_high':2,
    'wide_high':3
}
for col in ['first_option', 'second_option', 'chosen_type']:
    df[col] = df[col].map(dist_map)

# Build the arrays
for pid, group in df.groupby('participant_id'):
    pid_str = str(pid)
    globals()[f"{pid_str}_option1"] = group['first_option'].to_numpy()
    globals()[f"{pid_str}_option2"] = group['second_option'].to_numpy()
    globals()[f"{pid_str}_optionC"]  = group['chosen_type'].to_numpy()
    globals()[f"{pid_str}_confA"]    = group['action_selection_conf'].to_numpy()


# — assume `df` already has your mapped ints and full 108‐trial padding ——

# 1) Create two new columns where the smaller code is always in “opt1”
#    and the larger in “opt2”
df[['opt1','opt2']] = pd.DataFrame(
    np.sort(df[['first_option','second_option']].to_numpy(), axis=1),
    index=df.index
)

# 2) Now for each participant, get a frequency table of (opt1, opt2)
pair_counts = (
    df
    .groupby(['participant_id','opt1','opt2'])
    .size()
    .reset_index(name='count')
)

# 3) View the results
#    e.g. all unique pairs & counts for participant 217:
print(pair_counts[pair_counts['participant_id']==217])


