import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors

from inference import (
    predict_melting_point,
    predict_batch,
    explain_prediction
)

from hybrid_inference import (
    predict_hybrid_gat,
    get_hybrid_feature_importance,
    explain_hybrid_gat_prediction
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Melting Point AI Predictor",
    page_icon="🧪",
    layout="wide"
)


# =====================================================
# AUTHENTICATION
# =====================================================

with open("auth_config.yaml", "r") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"]
)

try:
    authenticator.login(
        location="main",
        fields={
            "Form name": "Login",
            "Username": "Username",
            "Password": "Password",
            "Login": "Login"
        }
    )
except TypeError:
    authenticator.login("Login", "main")


if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None


if st.session_state["authentication_status"] is True:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.success(
        f"Welcome {st.session_state.get('name', 'User')}"
    )

    # =====================================================
    # MAIN APP
    # =====================================================

    st.title("🧪 Melting Point Prediction AI")

    st.write(
        "Predict molecular melting point using RDKit descriptors, "
        "LightGBM, and Hybrid GAT AI."
    )

    tab1, tab2, tab3 = st.tabs([
        "Single SMILES Prediction",
        "Batch CSV Prediction",
        "Use Saved Full Dataset"
    ])

    # =====================================================
    # TAB 1 — SINGLE PREDICTION
    # =====================================================

    with tab1:

        st.subheader("Single Molecule Prediction")

        try:
            smiles_df = pd.read_csv("all_smiles_clean.csv")
            smiles_list = (
                smiles_df["SMILES"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_smiles = st.selectbox(
                "Select a SMILES molecule",
                smiles_list
            )

            manual_smiles = st.text_input(
                "Or Enter Custom SMILES",
                value=selected_smiles
            )

        except FileNotFoundError:
            st.warning("all_smiles_clean.csv not found. Manual input only.")
            manual_smiles = st.text_input("Enter SMILES", value="CCO")

        model_choice = st.radio(
            "Select Prediction Model",
            [
                "RDKit LightGBM",
                "Hybrid Descriptor + GAT"
            ]
        )

        mol = Chem.MolFromSmiles(manual_smiles)

        if mol is not None:

            molecule_image = Draw.MolToImage(
                mol,
                size=(400, 400)
            )

            st.image(
                molecule_image,
                caption="Molecular Structure"
            )

            properties_df = pd.DataFrame({
                "Property": [
                    "Molecular Formula",
                    "Molecular Weight",
                    "LogP",
                    "TPSA",
                    "H-Bond Donors",
                    "H-Bond Acceptors",
                    "Rotatable Bonds",
                    "Ring Count"
                ],
                "Value": [
                    rdMolDescriptors.CalcMolFormula(mol),
                    round(Descriptors.MolWt(mol), 2),
                    round(Descriptors.MolLogP(mol), 2),
                    round(Descriptors.TPSA(mol), 2),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol)
                ]
            })

            st.subheader("Molecular Properties")
            st.dataframe(properties_df)

        else:
            st.error("Invalid SMILES string.")

        if st.button("Predict Melting Point"):

            try:

                if model_choice == "RDKit LightGBM":
                    prediction_k = float(
                        predict_melting_point(manual_smiles)
                    )
                else:
                    prediction_k = float(
                        predict_hybrid_gat(manual_smiles)
                    )

                prediction_c = prediction_k - 273.15

                st.success("Prediction completed successfully")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Melting Point (Kelvin)",
                        value=f"{prediction_k:.2f} K"
                    )

                with col2:
                    st.metric(
                        label="Melting Point (Celsius)",
                        value=f"{prediction_c:.2f} °C"
                    )

                st.info(f"SMILES: {manual_smiles}")
                st.info(f"Model Used: {model_choice}")

                if model_choice == "RDKit LightGBM":

                    st.subheader("RDKit LightGBM SHAP Explanation")

                    explanation_df = explain_prediction(manual_smiles)
                    st.dataframe(explanation_df)

                    fig, ax = plt.subplots(figsize=(8, 5))

                    ax.barh(
                        explanation_df["Feature"],
                        explanation_df["SHAP_Value"]
                    )

                    ax.set_xlabel("SHAP Value")
                    ax.set_ylabel("Feature")
                    ax.set_title("Top RDKit LightGBM SHAP Contributions")

                    st.pyplot(fig)

                else:

                    st.subheader("Hybrid GAT SHAP Explanation")

                    hybrid_shap_df = explain_hybrid_gat_prediction(
                        manual_smiles,
                        top_n=10
                    )

                    st.dataframe(hybrid_shap_df)

                    fig, ax = plt.subplots(figsize=(8, 5))

                    ax.barh(
                        hybrid_shap_df["Feature"],
                        hybrid_shap_df["SHAP_Value"]
                    )

                    ax.set_xlabel("SHAP Value")
                    ax.set_ylabel("Feature")
                    ax.set_title("Top Hybrid GAT SHAP Contributions")

                    st.pyplot(fig)

                    st.subheader("Hybrid GAT Feature Importance")

                    hybrid_importance_df = get_hybrid_feature_importance(
                        top_n=15
                    )

                    st.dataframe(hybrid_importance_df)

                    fig2, ax2 = plt.subplots(figsize=(8, 5))

                    ax2.barh(
                        hybrid_importance_df["Feature"],
                        hybrid_importance_df["Importance"]
                    )

                    ax2.set_xlabel("Feature Importance")
                    ax2.set_ylabel("Feature")
                    ax2.set_title("Top Hybrid GAT Feature Importances")

                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # =====================================================
    # TAB 2 — BATCH CSV PREDICTION
    # =====================================================

    with tab2:

        st.subheader("Batch CSV Prediction")

        uploaded_file = st.file_uploader(
            "Upload CSV file with a column named SMILES",
            type=["csv"]
        )

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

            st.write("Uploaded Data Preview")
            st.dataframe(df.head())
            st.write(f"Total rows uploaded: {len(df)}")

            if "SMILES" not in df.columns:
                st.error("CSV must contain a column named SMILES")

            else:

                if st.button("Run Batch Prediction"):

                    results_df = predict_batch(
                        df["SMILES"]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

                    results_df["Predicted_Melting_Point_C"] = (
                        results_df["Predicted_Melting_Point_K"] - 273.15
                    ).round(2)

                    st.success("Batch prediction completed")
                    st.write(f"Total predictions: {len(results_df)}")
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="melting_point_predictions.csv",
                        mime="text/csv"
                    )

    # =====================================================
    # TAB 3 — SAVED FULL DATASET
    # =====================================================

    with tab3:

        st.subheader("Predict Saved Full Dataset")

        file_path = "all_smiles_clean.csv"

        try:

            full_df = pd.read_csv(file_path)

            st.write("Saved Dataset Preview")
            st.dataframe(full_df.head())
            st.write(f"Total rows found: {len(full_df)}")

            if "SMILES" not in full_df.columns:
                st.error("Saved dataset must contain a SMILES column")

            else:

                if st.button("Run Prediction on Saved Dataset"):

                    results_df = predict_batch(
                        full_df["SMILES"]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

                    results_df["Predicted_Melting_Point_C"] = (
                        results_df["Predicted_Melting_Point_K"] - 273.15
                    ).round(2)

                    st.success("Full dataset prediction completed")
                    st.write(f"Total predictions: {len(results_df)}")
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Full Dataset Predictions CSV",
                        data=csv,
                        file_name="all_smiles_predictions.csv",
                        mime="text/csv"
                    )

        except FileNotFoundError:
            st.warning(
                "all_smiles_clean.csv not found. "
                "Place it inside deployment folder."
            )

    st.markdown("---")

    st.caption(
        "Models: RDKit LightGBM + Hybrid Descriptor GAT | Secured with User Login"
    )


elif st.session_state["authentication_status"] is False:

    st.error("Incorrect username or password")


else:

    st.warning("Please enter username and password")