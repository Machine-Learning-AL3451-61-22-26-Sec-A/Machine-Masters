import pandas as pd
import numpy as np
import streamlit as st

def learn(concepts, target):
    specific_h = concepts.iloc[0].copy()
    st.write("\nInitialization of specific_h and general_h")
    st.write(specific_h)

    general_h = pd.DataFrame([["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))],
                             columns=specific_h.index)
    st.write(general_h)

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
                    
        st.write("\nSteps of Candidate Elimination Algorithm", i + 1)
        st.write(specific_h)
        st.write(general_h)

    general_h = general_h.loc[~(general_h == '?').all(axis=1)]
    return specific_h, general_h

def main():
    st.write("22AIA-MACHINE MASTERS")
    st.title("Candidate Elimination Algorithm")

    # File uploader for training data
    uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            concepts = data.iloc[:, :-1]
            target = data.iloc[:, -1]

            st.write("\nTraining Data:")
            st.write(data)

            s_final, g_final = learn(concepts, target)

            st.write("\nFinal Specific_h:")
            st.write(s_final)

            st.write("\nFinal General_h:")
            st.write(g_final)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
