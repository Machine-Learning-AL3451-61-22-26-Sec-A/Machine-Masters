import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("heartdisease.csv")
heart_disease = pd.DataFrame(data)
print("Loaded dataset:")
print(heart_disease.head())

# Define the feature and target columns
features = ['age', 'Gender', 'Family', 'diet', 'Lifestyle', 'cholestrol']
target = 'heartdisease'

# Ensure all categorical data are in string format to avoid issues with groupby
heart_disease[features] = heart_disease[features].astype(str)

# Calculate the prior probabilities P(heartdisease)
P_hd = heart_disease[target].value_counts(normalize=True)
print("\nPrior probabilities (P(heartdisease)):")
print(P_hd)

# Calculate the conditional probabilities P(feature | heartdisease)
P_features_given_hd = {}
for feature in features:
    P_features_given_hd[feature] = heart_disease.groupby(target)[feature].value_counts(normalize=True).unstack(fill_value=0)
    print(f"\nConditional probabilities P({feature} | heartdisease):")
    print(P_features_given_hd[feature])

# Function to calculate P(heartdisease | evidence)
# Function to calculate P(heartdisease | evidence)
def naive_bayes(evidence):
    # Initialize the result with zeros for each class label
    result = pd.Series(0, index=P_hd.index)
    for hd in result.index:
        for feature, value in evidence.items():
            if value in P_features_given_hd[feature].columns:
                result[hd] *= P_features_given_hd[feature].loc[hd, value]
            else:
                result[hd] *= 0
    # Normalize the result to get probabilities
    result /= result.sum()
    return result


# Input guidance
print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')
print('For Gender Enter { Male:0, Female:1 }')
print('For Family History Enter { yes:1, No:0 }')
print('For diet Enter { High:0, Medium:1 }')
print('For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }')
print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')

# Collect user input
evidence = {
    'age': str(input('Enter age: ')),
    'Gender': str(input('Enter Gender: ')),
    'Family': str(input('Enter Family history: ')),
    'diet': str(input('Enter diet: ')),
    'Lifestyle': str(input('Enter Lifestyle: ')),
    'cholestrol': str(input('Enter cholesterol: '))
}

# Perform inference
probabilities = naive_bayes(evidence)

# Output the result
print("\nProbability of heart disease:")
print(probabilities)
