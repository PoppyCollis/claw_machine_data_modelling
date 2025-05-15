import numpy as np
import pandas as pd

# --- Load and preprocess the CSV ---
data_path = 'data/all_participants_action_selection_data.csv'
cols = ['participant_id', 'trial_number', 'first_option', 'second_option', 'chosen_type', 'action_selection_conf']
df = pd.read_csv(data_path, usecols=cols)

# --- (Optional) pad missing trials for participant 217 only ---
out_dfs = []
for pid, sub in df.groupby('participant_id'):
    sub = sub.set_index('trial_number').sort_index()
    if pid == 217:
        sub = sub.reindex(range(1,109))    # insert NaNs for missing trials
    out_dfs.append(sub.assign(participant_id=pid).reset_index())
df = pd.concat(out_dfs, ignore_index=True)

# --- Map distribution-type strings to integers ---
dist_map = {
    'narrow_low':  0,
    'wide_low':    1,
    'narrow_high': 2,
    'wide_high':   3,
}
for col in ['first_option', 'second_option', 'chosen_type']:
    df[col] = df[col].map(dist_map)

# --- For each participant, create joint confidence-choice matrices per unique sorted pair ---
# We'll name each matrix '{participant_id}_{a}_{b}' where (a,b) is the sorted option code pair.
for pid, sub in df.groupby('participant_id'):
    pid_str = str(pid)
    # Compute sorted pairs
    sub = sub.assign(
        opt_min = np.minimum(sub['first_option'], sub['second_option']),
        opt_max = np.maximum(sub['first_option'], sub['second_option'])
    )
    grouped = sub.groupby(['opt_min', 'opt_max'])

    for (a, b), grp in grouped:
        total = len(grp)
        # Initialize joint count matrix: rows=confidence levels 1..6, cols=[choice_min, choice_max]
        joint_counts = np.zeros((6, 2), dtype=int)
        for lvl in range(1, 7):
            mask_lvl = grp['action_selection_conf'] == lvl
            joint_counts[lvl-1, 0] = np.sum(grp[mask_lvl]['chosen_type'] == a)
            joint_counts[lvl-1, 1] = np.sum(grp[mask_lvl]['chosen_type'] == b)
        # Convert counts to probabilities (joint distribution)
        joint_dist = joint_counts.astype(float) / total

        # Assign to a named variable in globals
        var_name = f"{pid_str}_{a}_{b}"
        globals()[var_name] = joint_dist

# After running this script, for each participant you'll have 6 new arrays:
#  - '<pid>_0_1', '<pid>_0_2', '<pid>_0_3', '<pid>1_2', '<pid>1_3', '<pid>2_3'
# each is a 6×2 matrix where:
#   - row i (0-indexed) is confidence level (i+1)
#   - col 0 = P(conf=i+1 AND choice=opt_min)
#   - col 1 = P(conf=i+1 AND choice=opt_max)
