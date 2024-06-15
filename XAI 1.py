#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.pipeline import Pipeline
from collections import Counter
import lime
import lime.lime_tabular
import xgboost as xgb
import seaborn as sns
import shap
shap.initjs()


# In[3]:


df = pd.read_csv("C:/Users/arjun/Downloads/Explainable_AI/Explainable-AI_Group-15/Loan_default.csv")
df.head()


# In[4]:


# One-hot encoding categorical variables
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded = df_encoded.head(1000)
df_encoded.head()


# In[5]:


# Define features and target variable
features = df_encoded.drop(columns=['LoanID', 'Default'])  
target = df_encoded['Default']

# Check the class distribution before sampling
print("Class distribution before sampling:", Counter(target))

# Determine the number of samples in the minority class
minority_class_samples = sum(target == 1)

# Define the sampling strategy
sampling_strategy = {1: minority_class_samples * 2}  # For example, 2 times the number of samples in the minority class for the majority class

# Define the oversampling and undersampling pipelines
over_sampling = RandomOverSampler(sampling_strategy=sampling_strategy)
under_sampling = RandomUnderSampler(sampling_strategy=sampling_strategy)

# Create the pipeline for resampling
resampling_pipeline = Pipeline([
    ('over_sampling', over_sampling),
    ('under_sampling', under_sampling)
])

# Apply the resampling pipeline to the data
X_resampled, y_resampled = resampling_pipeline.fit_resample(features, target)

# Check the class distribution after sampling
print("Class distribution after sampling:", Counter(y_resampled))


# In[6]:


# Select only the continuous variables from your DataFrame
continuous_vars = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = continuous_vars.corr()

# Plot the correlation matrix using seaborn's heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Matrix of Continuous Variables")
plt.show()


# ## Model Creation

# In[7]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[8]:


# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf_model.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred))


# ## LIME Explanations

# In[9]:


# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X_train.columns.tolist(),
                                                   class_names=['Non-Default', 'Default'],
                                                   discretize_continuous=True)

# Select an instance from the test set to explain
instance_idx = 0
instance = X_test.iloc[[instance_idx]].values[0]
explanation = explainer.explain_instance(instance,
                                         rf_model.predict_proba,
                                         num_features=len(X_train.columns))

# Print LIME explanation
print("LIME Explanation for instance", instance_idx)
print(explanation.as_list())



# In[10]:


# Extract feature names and LIME explanation values
feature_names = explanation.as_list()
weights = [val[1] for val in feature_names]
features = [val[0] for val in feature_names]

# Plot LIME explanation
plt.figure(figsize=(10, 8))
plt.barh(features, weights, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('LIME Explanation')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# ## Shap Explanations

# In[11]:


# Create the explainer
explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)


# In[12]:


shap_values = shap_values[:, :, 1]


# In[13]:


shap.summary_plot(shap_values, X_test)


# In[14]:


# Create an Explanation object using the SHAP values for the chosen instance
shap_explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=X_test.iloc[0])

# Plot the waterfall plot for the chosen instance
shap.plots.waterfall(shap_explanation)

shap.plots.force(shap_explanation)



# In[15]:


# Plot the dependence plot
shap.dependence_plot("InterestRate", shap_values, X_test, interaction_index='Income', show=False)

# Show the plot
plt.show()


# In[16]:


# Create an Explanation object using the SHAP values for the chosen instance
shap_explanation = shap.Explanation(values=shap_values[:100], base_values=explainer.expected_value[0], data=X_test.iloc[:100])

# Plot the stacked force plot for the selected instances
shap.plots.force(shap_explanation, show=True)


# In[ ]:





# In[ ]:




