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

# --- Standardize option pairs and compute aggregated metrics ---
results = {}
for pid, sub in df.groupby('participant_id'):
    pid_str = str(pid)
    # Ensure sorted pairs
    sub = sub.assign(
        opt_min = np.minimum(sub['first_option'], sub['second_option']),
        opt_max = np.maximum(sub['first_option'], sub['second_option'])
    )

    # Prepare containers
    pair_list = []
    count_list = []
    bern_list = []
    conf_list = []

    # Process each unique pair
    grouped = sub.groupby(['opt_min', 'opt_max'])
    for (a, b), grp in grouped:
        total = len(grp)
        # Bernoulli P(choice=min), P(choice=max)
        count_min = np.sum(grp['chosen_type'] == a)
        p_min = count_min / total if total > 0 else np.nan
        p_max = 1 - p_min if total > 0 else np.nan
        # Categorical distribution over confidence levels 1-6
        conf_counts = [np.sum(grp['action_selection_conf'] == lvl) for lvl in range(1,7)]
        conf_probs = np.array(conf_counts, dtype=float) / total if total > 0 else np.full(6, np.nan)

        pair_list.append((a, b))
        count_list.append(total)
        bern_list.append((p_min, p_max))
        conf_list.append(conf_probs)

    # Sort by pair
    sorted_idx = np.argsort(np.array(pair_list), axis=0)[:,0]  # sort by first then second
    pair_array = np.array(pair_list)[sorted_idx]
    counts = np.array(count_list)[sorted_idx]
    bern = np.array(bern_list)[sorted_idx]
    conf = np.vstack(conf_list)[sorted_idx]

    # Unpack pair_array
    opt1_pairs = pair_array[:,0]
    opt2_pairs = pair_array[:,1]

    # Assign to globals
    globals()[f"{pid_str}_option1_pairs"] = opt1_pairs
    globals()[f"{pid_str}_option2_pairs"] = opt2_pairs
    globals()[f"{pid_str}_pair_counts"]   = counts
    globals()[f"{pid_str}_optionC"]       = bern   # (n_pairs x 2) Bernoulli
    globals()[f"{pid_str}_confA"]         = conf   # (n_pairs x 6) Categorical

# After running this script, each participant has:
#  - '<pid>_option1_pairs' & '<pid>_option2_pairs': arrays of unique sorted option codes
#  - '<pid>_pair_counts': total presentations per pair
#  - '<pid>_optionC': one row per pair, [P(pick opt_min), P(pick opt_max)]
#  - '<pid>_confA': one row per pair, length-6 vector of P(confidence=1..6)
