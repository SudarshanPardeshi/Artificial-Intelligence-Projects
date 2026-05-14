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

from database_utils import (
    create_prediction_table,
    log_prediction,
    load_prediction_logs,
    clear_prediction_logs,
    delete_prediction_row
)

from pubchem_utils import name_to_smiles


st.set_page_config(
    page_title="Melting Point AI Predictor",
    page_icon="🧪",
    layout="wide"
)


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

    create_prediction_table()

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome {st.session_state.get('name', 'User')}")

    st.title("🧪 Hybrid GNN AI Cheminformatics Platform")

    st.write(
        "Predict molecular melting point using RDKit descriptors, "
        "LightGBM, Hybrid GAT AI, and molecule name search."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Molecule Prediction",
        "Batch CSV Prediction",
        "Use Saved Full Dataset",
        "Prediction History"
    ])

    with tab1:

        st.subheader("Single Molecule Prediction")

        input_mode = st.radio(
            "Choose Input Method",
            [
                "Select from Dataset",
                "Enter Custom SMILES",
                "Search by Molecule Name"
            ]
        )

        compound_name = ""
        manual_smiles = "CCO"

        if input_mode == "Select from Dataset":

            try:
                smiles_df = pd.read_csv("all_smiles_clean.csv")

                smiles_list = (
                    smiles_df["SMILES"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                manual_smiles = st.selectbox(
                    "Select a SMILES molecule",
                    smiles_list
                )

            except FileNotFoundError:
                st.warning("all_smiles_clean.csv not found. Manual input only.")
                manual_smiles = st.text_input("Enter SMILES", value="CCO")

        elif input_mode == "Enter Custom SMILES":

            manual_smiles = st.text_input(
                "Enter Custom SMILES",
                value="CCO"
            )

        else:

            compound_name = st.text_input(
                "Enter Molecule Name",
                value="ethanol"
            )

            if st.button("Convert Name to SMILES"):

                converted_smiles = name_to_smiles(compound_name)

                if converted_smiles:

                    st.session_state["converted_smiles"] = converted_smiles
                    st.session_state["compound_name"] = compound_name

                    st.success(f"Molecule Name: {compound_name}")
                    st.success(f"SMILES Found: {converted_smiles}")

                else:
                    st.error("Could not find SMILES for this molecule name.")

            manual_smiles = st.session_state.get(
                "converted_smiles",
                "CCO"
            )

            if "compound_name" in st.session_state:
                st.info(
                    f"Selected Molecule Name: {st.session_state['compound_name']}"
                )

            st.info(f"Current SMILES: {manual_smiles}")

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
                        "Melting Point (Kelvin)",
                        f"{prediction_k:.2f} K"
                    )

                with col2:
                    st.metric(
                        "Melting Point (Celsius)",
                        f"{prediction_c:.2f} °C"
                    )

                if input_mode == "Search by Molecule Name":
                    st.info(
                        f"Molecule Name: {st.session_state.get('compound_name', compound_name)}"
                    )

                st.info(f"SMILES: {manual_smiles}")
                st.info(f"Model Used: {model_choice}")

                log_prediction(
                    username=st.session_state.get("username", "unknown"),
                    smiles=manual_smiles,
                    model_used=model_choice,
                    prediction_k=prediction_k,
                    prediction_c=prediction_c,
                    status="Success"
                )

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

                log_prediction(
                    username=st.session_state.get("username", "unknown"),
                    smiles=manual_smiles,
                    model_used=model_choice,
                    prediction_k=None,
                    prediction_c=None,
                    status=f"Failed: {e}"
                )

    with tab2:

        st.subheader("Batch CSV Prediction")

        uploaded_file = st.file_uploader(
            "Upload CSV with SMILES column",
            type=["csv"]
        )

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

            st.dataframe(df.head())

            if "SMILES" not in df.columns:

                st.error("CSV must contain SMILES column")

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
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Predictions CSV",
                        data=csv,
                        file_name="melting_predictions.csv",
                        mime="text/csv"
                    )

    with tab3:

        st.subheader("Predict Saved Full Dataset")

        try:
            full_df = pd.read_csv("all_smiles_clean.csv")

            st.dataframe(full_df.head())

            if st.button("Run Prediction on Full Dataset"):

                results_df = predict_batch(
                    full_df["SMILES"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                results_df["Predicted_Melting_Point_C"] = (
                    results_df["Predicted_Melting_Point_K"] - 273.15
                ).round(2)

                st.success("Prediction completed")
                st.dataframe(results_df)

                csv = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Full Dataset Predictions",
                    data=csv,
                    file_name="all_smiles_predictions.csv",
                    mime="text/csv"
                )

        except FileNotFoundError:

            st.warning("all_smiles_clean.csv not found")

    with tab4:

        st.subheader("Prediction History")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Entire History"):

                clear_prediction_logs()
                st.success("All prediction history deleted")
                st.rerun()

        rows = load_prediction_logs()

        if len(rows) == 0:

            st.info("No prediction history available")

        else:

            history_df = pd.DataFrame(
                rows,
                columns=[
                    "ID",
                    "Username",
                    "SMILES",
                    "Model Used",
                    "Prediction K",
                    "Prediction °C",
                    "Status",
                    "Created At"
                ]
            )

            st.dataframe(history_df)

            st.subheader("Delete Selected Prediction")

            selected_id = st.selectbox(
                "Select Prediction ID to Delete",
                history_df["ID"].tolist()
            )

            if st.button("Delete Selected Row"):

                delete_prediction_row(selected_id)
                st.success(f"Prediction row {selected_id} deleted")
                st.rerun()

            csv = history_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download History CSV",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

    st.markdown("---")

    st.caption(
        "Hybrid GNN AI Cheminformatics Platform | "
        "Login + PubChem Name Search + SQLite Logging"
    )


elif st.session_state["authentication_status"] is False:

    st.error("Incorrect username or password")

else:

    st.warning("Please enter username and password")