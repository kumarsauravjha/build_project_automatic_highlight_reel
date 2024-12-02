# %%
#%%

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# %%
# %%

data = pd.read_csv("provided_data.csv", header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# %%
# %%

target = pd.read_csv('target.csv')

# %%
# %%

data.info()

# %%
# %%

data.isna().sum()

# %%
# %%

data.head()

# %%
# %%

data.shape

# %%
# %%

data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# %%
#%%

data.isna().sum()

# %%
# %%

data['effort'] = data['effort'].interpolate(method='linear')

# %%
# %%

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# %%
# %%

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# %%
# %%

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# %%
# %%

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort']
X = merged[features]
y = merged['value']

# %%
# %%

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
#%%

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

# %%
# %%

window_size = 100  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)

# %%
# %%

y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# %%
# %%

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# %%
# %%

# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# %%
# %%

'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)

class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

class_weight_dict = dict(zip(classes, class_weights))

# %%
# %%

# Initialize the Logistic Regression model with class weights
log_reg_model = LogisticRegression(class_weight=class_weight_dict, solver='liblinear', random_state=42)

# Train the model on the training data
log_reg_model.fit(X_train, y_train)

# %%
# Make predictions
y_pred3 = log_reg_model.predict(X_lagged)

# Evaluate the model
print(classification_report(y_lagged, y_pred3))

#%%
# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred3})
predictions_df.to_csv('predictions_all.csv', index=False)

# %%
y_lagged.shape

# %%
y_pred3.shape

# %%
frames_test

# %%
predictions_df = pd.DataFrame({'frame': frames_lagged, 'value': y_pred3})
predictions_df.to_csv('predictions_all.csv', index=False)


