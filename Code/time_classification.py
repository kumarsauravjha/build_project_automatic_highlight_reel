#%%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# %%
data = pd.read_csv("../Data/provided_data.csv", header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
# %%
target = pd.read_csv('../Data/target.csv')
# %%
data.info()

#%%
target.info()
# %%
data.isna().sum()
# %%
data.head()

#%%
target.head()
# %%
data.shape
# %%
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# %%
data['effort'] = data['effort'].interpolate(method='linear')

#%%
#creating custom features
data['speed'] = ((data['xc'].diff()**2 + data['yc'].diff()**2)**0.5)

data['acceleration'] = data['speed'].diff()
#%%
data.isna().sum()
# %%
# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)
# %%
# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)
# %%
# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

#%%
merged.to_csv("../Data/all_data_with_custom_features.csv")
# %%
# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort', 'speed', 'acceleration']
X = merged[features]
y = merged['value']
# %%
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
window_size = 100  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)
# %%
y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# %%
# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)
# %%
# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# %%
# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# %%
# Predict on the test set
y_pred = clf.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))

# Write predictions to CSV with the same syntax as target.csv
# predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
# predictions_df.to_csv('predictions.csv', index=False)

# %%
'''SVM'''

# from sklearn.svm import SVC
# from sklearn.metrics import classification_report

# # Initialize the SVM classifier
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # 'rbf' is the default kernel for non-linear separation

# # Train the model on the training data
# svm_model.fit(X_train, y_train)

# # Make predictions
# y_pred2 = svm_model.predict(X_test)

# # Evaluate the model
# print(classification_report(y_test, y_pred2))
# %%
'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)

class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

class_weight_dict = dict(zip(classes, class_weights))
# %%
# Initialize the Logistic Regression model with class weights
log_reg_model = LogisticRegression(class_weight=class_weight_dict, solver='liblinear', random_state=42)

# Train the model on the training data
log_reg_model.fit(X_train, y_train)

# Make predictions
y_pred3 = log_reg_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred3))

#%%
'''trying to improve recall of logistic regression'''
# Predict probabilities
y_pred_prob = log_reg_model.predict_proba(X_test)[:, 1]

# Lower threshold to 0.3 for higher recall
y_pred3 = (y_pred_prob > 0.3).astype(int)

# Evaluate the model
print(classification_report(y_test, y_pred3))

#%%
'''Hyperparameter tuning for the Logistic Regression to find best model'''
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'penalty': ['l2'],  # Regularization type
#     'solver': ['liblinear']  # Compatible solver
# }

# # Grid search
# grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=42),
#                            param_grid, scoring='recall', cv=5, verbose=1)
# grid_search.fit(X_train, y_train)

# # Best model
# best_model = grid_search.best_estimator_
# print(grid_search.best_params_)

# # Predict and evaluate
# y_pred = best_model.predict(X_test)
# print(classification_report(y_test, y_pred))

#%%
'''tweaking the cutoff for best model'''
# Predict probabilities
# y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# # Lower threshold to 0.3 for higher recall
# y_pred = (y_pred_prob > 0.3).astype(int)

# # Evaluate the model
# print(classification_report(y_test, y_pred))
# # %%
# '''XGBoost'''
# from xgboost import XGBClassifier
# scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# %%
# xgb_model = XGBClassifier(
#     objective='binary:logistic',
#     scale_pos_weight=scale_pos_weight,  # Address class imbalance
#     eval_metric='logloss',
#     # use_label_encoder=False,
#     random_state=42
# )

# # Train the model on the training data
# xgb_model.fit(X_train, y_train)

# # Make predictions
# y_pred = xgb_model.predict(X_test)

# # Evaluate the model
# print(classification_report(y_test, y_pred))
# %%
'''getting predictions for all the data'''

# Make predictions
y_pred_all = log_reg_model.predict(X_lagged)

# Evaluate the model
print(classification_report(y_lagged, y_pred_all))
# %%
predictions_df = pd.DataFrame({'frame': frames_lagged, 'value': y_pred_all})
predictions_df.to_csv('../Data/predictions_all_new.csv', index=False)
# %%
