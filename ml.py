import pandas as pd
import numpy as np

data = pd.read_csv("trainingdata.csv")
concepts = data.iloc[:, 0:-1]
target = data.iloc[:, -1]

def learn(concepts, target):
    specific_h = concepts.iloc[0].copy()
    print("\nInitialization of specific_h and general_h")
    print(specific_h)

    general_h = pd.DataFrame([["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))],
                              columns=specific_h.index)
    print(general_h)

    for i, h in concepts.iterrows():
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h.iloc[x, x] = '?'
        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h.iloc[x, x] = specific_h[x]
                else:
                    general_h.iloc[x, x] = '?'
        print("\nSteps of Candidate Elimination Algorithm", i + 1)
        print(specific_h)
        print(general_h)

    general_h = general_h.loc[~(general_h == '?').all(axis=1)]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:")
print(s_final)

print("\nFinal General_h:")
print(g_final)