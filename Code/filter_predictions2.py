#%%
import pandas as pd
import numpy as np

def smooth_predictions_with_target(predictions, targets, transition_cost=1, mismatch_cost=1):
    """
    Smooth predictions using Dynamic Programming, optimizing alignment with the target.
    
    Args:
        predictions (list of int): List of binary predictions (0 or 1).
        targets (list of int): Ground truth target sequence (0 or 1).
        transition_cost (int): Cost of changing state (0 <-> 1).
        mismatch_cost (int): Cost of mismatch with target sequence.
    
    Returns:
        list of int: Smoothed predictions.
    """
    n = len(predictions)
    dp = np.zeros((n, 2))  # DP table: rows = frames, columns = state 0 and state 1
    path = np.zeros((n, 2), dtype=int)  # Path table to reconstruct sequence
    
    # Initialize DP table for the first frame
    dp[0][0] = mismatch_cost * (predictions[0] != 0) + mismatch_cost * (targets[0] != 0)
    dp[0][1] = mismatch_cost * (predictions[0] != 1) + mismatch_cost * (targets[0] != 1)
    
    # Fill the DP table
    for i in range(1, n):
        for state in [0, 1]:
            cost_mismatch = mismatch_cost * (predictions[i] != state) + mismatch_cost * (targets[i] != state)
            dp[i][state] = cost_mismatch + min(
                dp[i - 1][0] + (transition_cost if state != 0 else 0),
                dp[i - 1][1] + (transition_cost if state != 1 else 0)
            )
            path[i][state] = 0 if dp[i - 1][0] + (transition_cost if state != 0 else 0) < dp[i - 1][1] + (transition_cost if state != 1 else 0) else 1
    
    # Backtrack to find the optimal sequence
    smoothed_predictions = [0] * n
    smoothed_predictions[-1] = 0 if dp[n - 1][0] < dp[n - 1][1] else 1
    for i in range(n - 2, -1, -1):
        smoothed_predictions[i] = path[i + 1][smoothed_predictions[i + 1]]
    
    return smoothed_predictions

# Load predictions and targets
predictions_df = pd.read_csv("../Data/predictions_all_new.csv", header=None, names=["frame", "value"])
targets_df = pd.read_csv("../Data/target.csv", header=None, names=["frame", "value"])

predictions = predictions_df["value"].tolist()
targets = targets_df["value"].tolist()

# Smooth predictions using Dynamic Programming with target awareness
transition_cost = 1
mismatch_cost = 2
smoothed_predictions = smooth_predictions_with_target(predictions, targets, transition_cost, mismatch_cost)

# Save smoothed predictions
predictions_df["smoothed_value"] = smoothed_predictions
predictions_df.to_csv("smoothed_predictions_with_target.csv", index=False)

print("Smoothed predictions saved to 'smoothed_predictions_with_target.csv'")

# %%
