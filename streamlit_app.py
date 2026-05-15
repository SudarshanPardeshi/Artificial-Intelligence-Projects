
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
from io import BytesIO
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from inference import predict_melting_point, predict_batch, explain_prediction
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


# ==================================================
# AUTHENTICATION
# ==================================================

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


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def load_molecule_dataset():

    possible_files = [
        "all_smiles_with_names.csv",
        "all_smiles_clean.csv",
        "test_smiles_with_names.csv"
    ]

    for file_path in possible_files:

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        # Standardize SMILES column
        if "SMILES" not in df.columns:

            possible_smiles_cols = [
                "smiles",
                "Smiles",
                "canonical_smiles",
                "Canonical_SMILES"
            ]

            found_smiles_col = None

            for col in possible_smiles_cols:
                if col in df.columns:
                    found_smiles_col = col
                    break

            if found_smiles_col is None:
                continue

            df = df.rename(
                columns={
                    found_smiles_col: "SMILES"
                }
            )

        # Standardize molecule name column
        if "Molecule_Name" not in df.columns:

            if "IUPACName" in df.columns:

                df["Molecule_Name"] = df["IUPACName"]

            elif "IUPAC_Name" in df.columns:

                df["Molecule_Name"] = df["IUPAC_Name"]

            elif "Name" in df.columns:

                df["Molecule_Name"] = df["Name"]

            elif "name" in df.columns:

                df["Molecule_Name"] = df["name"]

            elif "Molecule" in df.columns:

                df["Molecule_Name"] = df["Molecule"]

            else:

                df["Molecule_Name"] = "Name Not Found"

        df = df.dropna(
            subset=[
                "SMILES"
            ]
        ).copy()

        df["SMILES"] = df["SMILES"].astype(str).str.strip()

        df = df[
            df["SMILES"] != ""
        ].copy()

        df["Molecule_Name"] = df["Molecule_Name"].fillna(
            "Name Not Found"
        )

        df["Molecule_Name"] = df["Molecule_Name"].astype(str)

        df = df.drop_duplicates(
            subset=[
                "SMILES"
            ]
        ).reset_index(drop=True)

        df["Molecule_Display"] = (
            df["Molecule_Name"].astype(str)
            +
            " | "
            +
            df["SMILES"].astype(str)
        )

        df["Dataset_Source_File"] = file_path

        return df.reset_index(drop=True)

    raise FileNotFoundError(
        "No valid molecule dataset found. Please add all_smiles_with_names.csv "
        "to the same folder as streamlit_app.py."
    )


def display_paginated_molecule_table(
    df,
    table_key,
    rows_per_page=100,
    columns=None
):
    if columns is None:
        columns = [
            "Molecule_Name",
            "SMILES"
        ]

    if df is None or df.empty:
        st.warning("No molecules available to display.")
        return df

    safe_columns = [
        col for col in columns
        if col in df.columns
    ]

    if not safe_columns:
        st.warning("No display columns found in this dataframe.")
        return df

    total_rows = len(df)

    st.markdown("#### Molecule Catalog Pagination")

    col_page_size, col_page_number = st.columns([1, 1])

    with col_page_size:

        selected_rows_per_page = st.selectbox(
            "Rows per page",
            options=[
                25,
                50,
                100,
                200,
                500
            ],
            index=2,
            key=f"{table_key}_rows_per_page"
        )

    total_pages = max(
        1,
        math.ceil(total_rows / selected_rows_per_page)
    )

    with col_page_number:

        page_number = st.number_input(
            "Page number",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{table_key}_page_number"
        )

    start_idx = (
        page_number - 1
    ) * selected_rows_per_page

    end_idx = start_idx + selected_rows_per_page

    paged_df = df.iloc[
        start_idx:end_idx
    ].copy()

    st.dataframe(
        paged_df[safe_columns],
        width="stretch"
    )

    st.caption(
        f"Showing molecules {start_idx + 1} to "
        f"{min(end_idx, total_rows)} out of {total_rows} "
        f"(Page {page_number} of {total_pages})"
    )

    return paged_df


def make_safe_filename(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace(";", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("[", "_")
        .replace("]", "_")
    )


def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )


def find_top_similar_molecules(query_smiles, molecule_df, top_n=10):
    query_fp = get_morgan_fingerprint(query_smiles)

    if query_fp is None:
        return pd.DataFrame()

    results = []

    for _, row in molecule_df.iterrows():
        candidate_smiles = row["SMILES"]

        if candidate_smiles == query_smiles:
            continue

        candidate_fp = get_morgan_fingerprint(candidate_smiles)

        if candidate_fp is None:
            continue

        similarity = DataStructs.TanimotoSimilarity(
            query_fp,
            candidate_fp
        )

        results.append({
            "IUPAC_Name": row["Molecule_Name"],
            "SMILES": candidate_smiles,
            "Similarity_Score": round(similarity, 4)
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    return results_df.sort_values(
        by="Similarity_Score",
        ascending=False
    ).head(top_n)


def create_3d_molblock(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.maxAttempts = 1000

        result = AllChem.EmbedMolecule(
            mol,
            params
        )

        if result != 0:
            result = AllChem.EmbedMolecule(
                mol,
                randomSeed=42,
                useRandomCoords=True,
                maxAttempts=1000
            )

        if result != 0:
            return None

        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    maxIters=500
                )
            else:
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=500
                )

        except Exception:
            pass

        return Chem.MolToMolBlock(mol)

    except Exception:
        return None


def show_3d_molecule(smiles, width=430, height=420, viewer_key=None):

    if not PY3DMOL_AVAILABLE:
        st.warning(
            "py3Dmol is not installed. Please run: pip install py3Dmol"
        )
        return

    mol_block = create_3d_molblock(smiles)

    if mol_block is None:
        st.warning(
            "3D structure could not be generated for this molecule. "
            "This can happen for some complex, very large, or invalid SMILES."
        )
        return

    try:
        viewer = py3Dmol.view(
            width=width,
            height=height
        )

        viewer.addModel(
            mol_block,
            "mol"
        )

        viewer.setStyle({
            "stick": {
                "radius": 0.18
            },
            "sphere": {
                "scale": 0.25
            }
        })

        viewer.setBackgroundColor("white")
        viewer.zoomTo()

        html = viewer._make_html()

        components.html(
            html,
            height=height + 40,
            width=width + 40,
            scrolling=False
        )

    except Exception as e:
        st.error(f"3D viewer rendering failed: {e}")


def calculate_prediction_uncertainty(rdkit_prediction, hybrid_prediction):
    difference = abs(rdkit_prediction - hybrid_prediction)
    confidence = max(0, 100 - difference)
    confidence = min(100, round(confidence, 2))
    uncertainty_range = round(difference / 2, 2)

    if confidence >= 85:
        confidence_label = "High Confidence"
    elif confidence >= 70:
        confidence_label = "Moderate Confidence"
    else:
        confidence_label = "Low Confidence"

    return {
        "difference": round(difference, 2),
        "confidence": confidence,
        "uncertainty_range": uncertainty_range,
        "confidence_label": confidence_label
    }


def extract_confidence_from_status(status):
    try:
        match = re.search(r"Confidence:\s*([0-9.]+)%", str(status))
        if match:
            return float(match.group(1))
        return None
    except Exception:
        return None


def clean_shap_dataframe(shap_df, top_n=10):
    if shap_df is None or shap_df.empty:
        return pd.DataFrame()

    temp_df = shap_df.copy()

    if "SHAP_Value" in temp_df.columns:
        temp_df["Abs_SHAP"] = temp_df["SHAP_Value"].abs()
        temp_df = temp_df.sort_values(
            by="Abs_SHAP",
            ascending=False
        )

    return temp_df.head(top_n)


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            return "No Scaffold"

        return Chem.MolToSmiles(scaffold_mol)

    except Exception:
        return None


def generate_scaffold_dataframe(molecule_df):
    scaffold_rows = []

    for _, row in molecule_df.iterrows():
        scaffold_rows.append({
            "Molecule_Name": row["Molecule_Name"],
            "SMILES": row["SMILES"],
            "Murcko_Scaffold": get_murcko_scaffold(row["SMILES"])
        })

    scaffold_df = pd.DataFrame(scaffold_rows)
    scaffold_df = scaffold_df.dropna(subset=["Murcko_Scaffold"])

    return scaffold_df



def calculate_ood_status(max_similarity):

    if max_similarity is None:

        return {
            "OOD_Status": "Unknown",
            "Reliability": "Unknown",
            "Warning": "OOD detection could not be completed."
        }

    if max_similarity >= 0.70:

        return {
            "OOD_Status": "In Distribution",
            "Reliability": "Reliable",
            "Warning": (
                "Molecule is chemically similar "
                "to known dataset molecules."
            )
        }

    elif max_similarity >= 0.40:

        return {
            "OOD_Status": "Borderline",
            "Reliability": "Use With Caution",
            "Warning": (
                "Molecule is moderately similar "
                "to dataset molecules."
            )
        }

    else:

        return {
            "OOD_Status": "Out of Distribution",
            "Reliability": "Prediction May Be Unreliable",
            "Warning": (
                "Molecule is unlike known dataset chemistry. "
                "Prediction may be unreliable."
            )
        }


def detect_ood_molecule(query_smiles, molecule_df):

    query_fp = get_morgan_fingerprint(query_smiles)

    if query_fp is None:

        return {
            "Query_SMILES": query_smiles,
            "Nearest_Molecule_Name": None,
            "Nearest_SMILES": None,
            "Max_Tanimoto_Similarity": None,
            "OOD_Status": "Invalid SMILES",
            "Reliability": "Invalid Input",
            "Warning": "Invalid SMILES string."
        }

    best_similarity = -1
    best_name = None
    best_smiles = None

    for _, row in molecule_df.iterrows():

        candidate_smiles = row["SMILES"]

        candidate_fp = get_morgan_fingerprint(
            candidate_smiles
        )

        if candidate_fp is None:
            continue

        similarity = DataStructs.TanimotoSimilarity(
            query_fp,
            candidate_fp
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_name = row["Molecule_Name"]
            best_smiles = candidate_smiles

    if best_similarity < 0:

        max_similarity = None
        status_info = calculate_ood_status(None)

    else:

        max_similarity = round(best_similarity, 4)
        status_info = calculate_ood_status(max_similarity)

    return {
        "Query_SMILES": query_smiles,
        "Nearest_Molecule_Name": best_name,
        "Nearest_SMILES": best_smiles,
        "Max_Tanimoto_Similarity": max_similarity,
        "OOD_Status": status_info["OOD_Status"],
        "Reliability": status_info["Reliability"],
        "Warning": status_info["Warning"]
    }



def smiles_to_morgan_array(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def generate_pca_chemical_space(molecule_df, n_bits=2048):
    rows = []
    fps = []

    for _, row in molecule_df.iterrows():

        smiles = row["SMILES"]
        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fps.append(fp_array)

        rows.append({
            "Molecule_Name": row["Molecule_Name"],
            "SMILES": smiles
        })

    if len(fps) < 2:
        return pd.DataFrame()

    fp_matrix = np.array(fps)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(fp_scaled)

    pca_df = pd.DataFrame(rows)
    pca_df["PCA1"] = pca_result[:, 0]
    pca_df["PCA2"] = pca_result[:, 1]

    explained_variance = pca.explained_variance_ratio_

    pca_df["PCA1_Explained_Variance"] = round(
        explained_variance[0] * 100,
        2
    )

    pca_df["PCA2_Explained_Variance"] = round(
        explained_variance[1] * 100,
        2
    )

    return pca_df



def generate_tsne_chemical_space(
    molecule_df,
    n_bits=2048,
    perplexity=30,
    random_state=42
):
    rows = []
    fps = []

    for _, row in molecule_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fps.append(fp_array)

        rows.append({
            "Molecule_Name": row["Molecule_Name"],
            "SMILES": smiles
        })

    if len(fps) < 5:
        return pd.DataFrame()

    fp_matrix = np.array(fps)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    safe_perplexity = min(
        int(perplexity),
        max(2, len(fp_scaled) - 1)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=int(random_state)
    )

    tsne_result = tsne.fit_transform(fp_scaled)

    tsne_df = pd.DataFrame(rows)
    tsne_df["TSNE1"] = tsne_result[:, 0]
    tsne_df["TSNE2"] = tsne_result[:, 1]
    tsne_df["Perplexity"] = safe_perplexity

    return tsne_df



def generate_umap_chemical_space(
    molecule_df,
    n_bits=2048,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
):
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is not installed. Run: pip install umap-learn"
        )

    rows = []
    fps = []

    for _, row in molecule_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fps.append(fp_array)

        rows.append({
            "Molecule_Name": row["Molecule_Name"],
            "SMILES": smiles
        })

    if len(fps) < 5:
        return pd.DataFrame()

    fp_matrix = np.array(fps)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    safe_neighbors = min(
        int(n_neighbors),
        max(2, len(fp_scaled) - 1)
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state)
    )

    umap_result = reducer.fit_transform(fp_scaled)

    umap_df = pd.DataFrame(rows)
    umap_df["UMAP1"] = umap_result[:, 0]
    umap_df["UMAP2"] = umap_result[:, 1]
    umap_df["n_neighbors"] = safe_neighbors
    umap_df["min_dist"] = float(min_dist)

    return umap_df



def calculate_all_similarity_scores(query_smiles, molecule_df):

    query_fp = get_morgan_fingerprint(query_smiles)

    if query_fp is None:
        return pd.DataFrame()

    similarity_rows = []

    for _, row in molecule_df.iterrows():

        candidate_name = row["Molecule_Name"]
        candidate_smiles = row["SMILES"]

        candidate_fp = get_morgan_fingerprint(candidate_smiles)

        if candidate_fp is None:
            continue

        similarity = DataStructs.TanimotoSimilarity(
            query_fp,
            candidate_fp
        )

        similarity_rows.append({
            "Molecule_Name": candidate_name,
            "SMILES": candidate_smiles,
            "Tanimoto_Similarity": round(similarity, 4)
        })

    similarity_df = pd.DataFrame(similarity_rows)

    if similarity_df.empty:
        return similarity_df

    similarity_df = similarity_df.sort_values(
        by="Tanimoto_Similarity",
        ascending=False
    ).reset_index(drop=True)

    return similarity_df



def calculate_mahalanobis_distance_for_smiles(
    query_smiles,
    molecule_df,
    max_reference_molecules=1000,
    n_bits=2048
):
    query_array = smiles_to_morgan_array(
        query_smiles,
        n_bits=n_bits
    )

    if query_array is None:
        return None, None, None

    reference_arrays = []

    reference_df = molecule_df.head(
        min(max_reference_molecules, len(molecule_df))
    ).copy()

    for smiles in reference_df["SMILES"]:

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is not None:
            reference_arrays.append(fp_array)

    if len(reference_arrays) < 10:
        return None, None, None

    reference_matrix = np.array(reference_arrays)

    try:
        pca_model = PCA(
            n_components=min(20, reference_matrix.shape[0] - 1),
            random_state=42
        )

        reference_latent = pca_model.fit_transform(
            reference_matrix
        )

        query_latent = pca_model.transform(
            np.array([query_array])
        )[0]

        covariance_model = EmpiricalCovariance().fit(
            reference_latent
        )

        mean_vector = covariance_model.location_
        precision_matrix = covariance_model.precision_

        diff = query_latent - mean_vector

        mahalanobis_distance = float(
            np.sqrt(
                diff.T @ precision_matrix @ diff
            )
        )

        reference_distances = []

        for row in reference_latent:
            row_diff = row - mean_vector
            row_distance = float(
                np.sqrt(
                    row_diff.T @ precision_matrix @ row_diff
                )
            )
            reference_distances.append(row_distance)

        reference_distances = np.array(reference_distances)

        threshold_95 = float(
            np.percentile(reference_distances, 95)
        )

        threshold_99 = float(
            np.percentile(reference_distances, 99)
        )

        return (
            round(mahalanobis_distance, 4),
            round(threshold_95, 4),
            round(threshold_99, 4)
        )

    except Exception:
        return None, None, None



def generate_ood_chemical_space_embeddings(
    query_smiles,
    molecule_df,
    max_reference_molecules=1000,
    n_bits=2048,
    method="PCA"
):
    query_fp_array = smiles_to_morgan_array(
        query_smiles,
        n_bits=n_bits
    )

    if query_fp_array is None:
        return pd.DataFrame()

    rows = []
    fingerprint_arrays = []

    reference_df = molecule_df.head(
        min(max_reference_molecules, len(molecule_df))
    ).copy()

    for _, row in reference_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fingerprint_arrays.append(fp_array)

        rows.append({
            "Molecule_Name": row["Molecule_Name"],
            "SMILES": smiles,
            "Point_Type": "Dataset"
        })

    if len(fingerprint_arrays) < 5:
        return pd.DataFrame()

    fingerprint_arrays.append(query_fp_array)

    rows.append({
        "Molecule_Name": "Query Molecule",
        "SMILES": query_smiles,
        "Point_Type": "Query"
    })

    fp_matrix = np.array(fingerprint_arrays)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    if method == "PCA":

        reducer = PCA(
            n_components=2,
            random_state=42
        )

        embedding = reducer.fit_transform(fp_scaled)

        result_df = pd.DataFrame(rows)
        result_df["X"] = embedding[:, 0]
        result_df["Y"] = embedding[:, 1]
        result_df["Method"] = "PCA"
        result_df["Explained_Variance_PC1"] = round(
            reducer.explained_variance_ratio_[0] * 100,
            2
        )
        result_df["Explained_Variance_PC2"] = round(
            reducer.explained_variance_ratio_[1] * 100,
            2
        )

        return result_df

    elif method == "UMAP":

        if not UMAP_AVAILABLE:
            raise ImportError(
                "umap-learn is not installed. Run: pip install umap-learn"
            )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )

        embedding = reducer.fit_transform(fp_scaled)

        result_df = pd.DataFrame(rows)
        result_df["X"] = embedding[:, 0]
        result_df["Y"] = embedding[:, 1]
        result_df["Method"] = "UMAP"

        return result_df

    else:
        return pd.DataFrame()



def calculate_deep_ensemble_uncertainty(
    rdkit_prediction,
    hybrid_prediction,
    ensemble_prediction
):
    predictions = np.array([
        rdkit_prediction,
        hybrid_prediction,
        ensemble_prediction
    ], dtype=float)

    mean_prediction = float(np.mean(predictions))
    std_prediction = float(np.std(predictions))
    min_prediction = float(np.min(predictions))
    max_prediction = float(np.max(predictions))
    prediction_range = float(max_prediction - min_prediction)

    if std_prediction <= 5:
        uncertainty_label = "Low Uncertainty"
        uncertainty_status = "Reliable"
    elif std_prediction <= 15:
        uncertainty_label = "Moderate Uncertainty"
        uncertainty_status = "Use With Caution"
    else:
        uncertainty_label = "High Uncertainty"
        uncertainty_status = "Prediction May Be Unstable"

    return {
        "Mean_Prediction_K": round(mean_prediction, 2),
        "STD_Prediction_K": round(std_prediction, 2),
        "Min_Prediction_K": round(min_prediction, 2),
        "Max_Prediction_K": round(max_prediction, 2),
        "Prediction_Range_K": round(prediction_range, 2),
        "Uncertainty_Label": uncertainty_label,
        "Uncertainty_Status": uncertainty_status
    }


def calculate_conformal_prediction_interval(
    prediction_k,
    uncertainty_range,
    confidence_label
):
    if confidence_label == "High Confidence":
        conformal_margin = max(10, uncertainty_range * 1.5)
        interval_label = "Narrow Interval"
    elif confidence_label == "Moderate Confidence":
        conformal_margin = max(20, uncertainty_range * 2.0)
        interval_label = "Moderate Interval"
    else:
        conformal_margin = max(30, uncertainty_range * 2.5)
        interval_label = "Wide Interval"

    lower_k = prediction_k - conformal_margin
    upper_k = prediction_k + conformal_margin

    return {
        "Lower_Bound_K": round(lower_k, 2),
        "Upper_Bound_K": round(upper_k, 2),
        "Lower_Bound_C": round(lower_k - 273.15, 2),
        "Upper_Bound_C": round(upper_k - 273.15, 2),
        "Conformal_Margin_K": round(conformal_margin, 2),
        "Interval_Label": interval_label
    }



def generate_interactive_pca_chemical_space(
    molecule_df,
    selected_smiles=None,
    sample_size=1000,
    random_state=42,
    n_bits=2048
):
    if molecule_df.empty:
        return pd.DataFrame(), None

    if sample_size < len(molecule_df):
        sample_df = molecule_df.sample(
            n=sample_size,
            random_state=random_state
        ).copy()
    else:
        sample_df = molecule_df.copy()

    if selected_smiles is not None and selected_smiles not in sample_df["SMILES"].values:
        selected_row_df = molecule_df[
            molecule_df["SMILES"] == selected_smiles
        ].copy()

        if not selected_row_df.empty:
            sample_df = pd.concat(
                [
                    sample_df,
                    selected_row_df
                ],
                ignore_index=True
            ).drop_duplicates(
                subset=["SMILES"]
            )

    fp_arrays = []
    valid_rows = []

    for _, row in sample_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fp_arrays.append(fp_array)
        valid_rows.append(row)

    if len(fp_arrays) < 5:
        return pd.DataFrame(), None

    fp_matrix = np.array(fp_arrays)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    pca_model = PCA(
        n_components=2,
        random_state=random_state
    )

    pca_embedding = pca_model.fit_transform(
        fp_scaled
    )

    pca_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    pca_df["PCA_1"] = pca_embedding[:, 0]
    pca_df["PCA_2"] = pca_embedding[:, 1]
    pca_df["Explained_Variance_PC1_%"] = round(
        pca_model.explained_variance_ratio_[0] * 100,
        2
    )
    pca_df["Explained_Variance_PC2_%"] = round(
        pca_model.explained_variance_ratio_[1] * 100,
        2
    )

    pca_df["Point_Type"] = "Dataset"

    if selected_smiles is not None:
        pca_df.loc[
            pca_df["SMILES"] == selected_smiles,
            "Point_Type"
        ] = "Selected Molecule"

    # Scaffold grouping
    pca_df["Murcko_Scaffold"] = pca_df["SMILES"].apply(
        get_murcko_scaffold
    )

    scaffold_counts = pca_df["Murcko_Scaffold"].value_counts()
    top_scaffolds = scaffold_counts.head(10).index.tolist()

    pca_df["Scaffold_Group"] = pca_df["Murcko_Scaffold"].apply(
        lambda x: x if x in top_scaffolds else "Other"
    )

    # PCA distance-based outlier score
    pca_center = np.array([
        pca_df["PCA_1"].mean(),
        pca_df["PCA_2"].mean()
    ])

    pca_df["PCA_Distance_From_Center"] = pca_df.apply(
        lambda row: float(
            np.sqrt(
                (row["PCA_1"] - pca_center[0]) ** 2
                +
                (row["PCA_2"] - pca_center[1]) ** 2
            )
        ),
        axis=1
    )

    outlier_threshold = pca_df[
        "PCA_Distance_From_Center"
    ].quantile(0.95)

    pca_df["Outlier_Status"] = pca_df[
        "PCA_Distance_From_Center"
    ].apply(
        lambda x: "Potential Outlier" if x >= outlier_threshold else "Normal"
    )

    # OOD-like status based on distance percentile
    pca_df["PCA_OOD_Status"] = pca_df[
        "PCA_Distance_From_Center"
    ].apply(
        lambda x: (
            "High Distance / Possible OOD"
            if x >= outlier_threshold
            else "Inside PCA Space"
        )
    )

    return pca_df, pca_model



def generate_interactive_tsne_chemical_space(
    molecule_df,
    selected_smiles=None,
    sample_size=1000,
    random_state=42,
    n_bits=2048,
    perplexity=30,
    learning_rate="auto",
    max_iter=1000
):
    if molecule_df.empty:
        return pd.DataFrame()

    if sample_size < len(molecule_df):
        sample_df = molecule_df.sample(
            n=sample_size,
            random_state=random_state
        ).copy()
    else:
        sample_df = molecule_df.copy()

    if selected_smiles is not None and selected_smiles not in sample_df["SMILES"].values:
        selected_row_df = molecule_df[
            molecule_df["SMILES"] == selected_smiles
        ].copy()

        if not selected_row_df.empty:
            sample_df = pd.concat(
                [
                    sample_df,
                    selected_row_df
                ],
                ignore_index=True
            ).drop_duplicates(
                subset=["SMILES"]
            )

    fp_arrays = []
    valid_rows = []

    for _, row in sample_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fp_arrays.append(fp_array)
        valid_rows.append(row)

    if len(fp_arrays) < 10:
        return pd.DataFrame()

    fp_matrix = np.array(fp_arrays)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    # t-SNE works better and faster after PCA pre-reduction
    pca_components = min(50, fp_scaled.shape[0] - 1, fp_scaled.shape[1])

    pca_pre = PCA(
        n_components=pca_components,
        random_state=random_state
    )

    fp_reduced = pca_pre.fit_transform(fp_scaled)

    safe_perplexity = min(
        perplexity,
        max(5, (fp_reduced.shape[0] - 1) // 3)
    )

    tsne_model = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=random_state
    )

    tsne_embedding = tsne_model.fit_transform(
        fp_reduced
    )

    tsne_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    tsne_df["TSNE_1"] = tsne_embedding[:, 0]
    tsne_df["TSNE_2"] = tsne_embedding[:, 1]
    tsne_df["Point_Type"] = "Dataset"

    if selected_smiles is not None:
        tsne_df.loc[
            tsne_df["SMILES"] == selected_smiles,
            "Point_Type"
        ] = "Selected Molecule"

    tsne_df["Murcko_Scaffold"] = tsne_df["SMILES"].apply(
        get_murcko_scaffold
    )

    scaffold_counts = tsne_df["Murcko_Scaffold"].value_counts()
    top_scaffolds = scaffold_counts.head(10).index.tolist()

    tsne_df["Scaffold_Group"] = tsne_df["Murcko_Scaffold"].apply(
        lambda x: x if x in top_scaffolds else "Other"
    )

    tsne_center = np.array([
        tsne_df["TSNE_1"].mean(),
        tsne_df["TSNE_2"].mean()
    ])

    tsne_df["TSNE_Distance_From_Center"] = tsne_df.apply(
        lambda row: float(
            np.sqrt(
                (row["TSNE_1"] - tsne_center[0]) ** 2
                +
                (row["TSNE_2"] - tsne_center[1]) ** 2
            )
        ),
        axis=1
    )

    outlier_threshold = tsne_df[
        "TSNE_Distance_From_Center"
    ].quantile(0.95)

    tsne_df["TSNE_Outlier_Status"] = tsne_df[
        "TSNE_Distance_From_Center"
    ].apply(
        lambda x: "Potential Outlier" if x >= outlier_threshold else "Normal"
    )

    tsne_df["TSNE_Chemical_Space_Status"] = tsne_df[
        "TSNE_Distance_From_Center"
    ].apply(
        lambda x: (
            "High Distance / Possible Isolated Molecule"
            if x >= outlier_threshold
            else "Inside t-SNE Neighborhood Space"
        )
    )

    tsne_df["tSNE_Perplexity_Used"] = safe_perplexity

    return tsne_df



def generate_interactive_umap_chemical_space(
    molecule_df,
    selected_smiles=None,
    sample_size=1000,
    random_state=42,
    n_bits=2048,
    n_neighbors=15,
    min_dist=0.1
):
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is not installed. Run: pip install umap-learn"
        )

    if molecule_df.empty:
        return pd.DataFrame()

    if sample_size < len(molecule_df):
        sample_df = molecule_df.sample(
            n=sample_size,
            random_state=random_state
        ).copy()
    else:
        sample_df = molecule_df.copy()

    if selected_smiles is not None and selected_smiles not in sample_df["SMILES"].values:
        selected_row_df = molecule_df[
            molecule_df["SMILES"] == selected_smiles
        ].copy()

        if not selected_row_df.empty:
            sample_df = pd.concat(
                [
                    sample_df,
                    selected_row_df
                ],
                ignore_index=True
            ).drop_duplicates(
                subset=["SMILES"]
            )

    fp_arrays = []
    valid_rows = []

    for _, row in sample_df.iterrows():

        smiles = row["SMILES"]

        fp_array = smiles_to_morgan_array(
            smiles,
            n_bits=n_bits
        )

        if fp_array is None:
            continue

        fp_arrays.append(fp_array)
        valid_rows.append(row)

    if len(fp_arrays) < 10:
        return pd.DataFrame()

    fp_matrix = np.array(fp_arrays)

    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_matrix)

    safe_neighbors = min(
        int(n_neighbors),
        max(2, len(fp_scaled) - 1)
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state)
    )

    umap_embedding = reducer.fit_transform(
        fp_scaled
    )

    umap_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    umap_df["UMAP_1"] = umap_embedding[:, 0]
    umap_df["UMAP_2"] = umap_embedding[:, 1]
    umap_df["Point_Type"] = "Dataset"

    if selected_smiles is not None:
        umap_df.loc[
            umap_df["SMILES"] == selected_smiles,
            "Point_Type"
        ] = "Selected Molecule"

    umap_df["Murcko_Scaffold"] = umap_df["SMILES"].apply(
        get_murcko_scaffold
    )

    scaffold_counts = umap_df["Murcko_Scaffold"].value_counts()
    top_scaffolds = scaffold_counts.head(10).index.tolist()

    umap_df["Scaffold_Group"] = umap_df["Murcko_Scaffold"].apply(
        lambda x: x if x in top_scaffolds else "Other"
    )

    umap_center = np.array([
        umap_df["UMAP_1"].mean(),
        umap_df["UMAP_2"].mean()
    ])

    umap_df["UMAP_Distance_From_Center"] = umap_df.apply(
        lambda row: float(
            np.sqrt(
                (row["UMAP_1"] - umap_center[0]) ** 2
                +
                (row["UMAP_2"] - umap_center[1]) ** 2
            )
        ),
        axis=1
    )

    outlier_threshold = umap_df[
        "UMAP_Distance_From_Center"
    ].quantile(0.95)

    umap_df["UMAP_Outlier_Status"] = umap_df[
        "UMAP_Distance_From_Center"
    ].apply(
        lambda x: "Potential Outlier" if x >= outlier_threshold else "Normal"
    )

    umap_df["UMAP_Chemical_Space_Status"] = umap_df[
        "UMAP_Distance_From_Center"
    ].apply(
        lambda x: (
            "High Distance / Possible Isolated Molecule"
            if x >= outlier_threshold
            else "Inside UMAP Neighborhood Space"
        )
    )

    umap_df["UMAP_n_neighbors_Used"] = safe_neighbors
    umap_df["UMAP_min_dist_Used"] = float(min_dist)

    return umap_df



def calculate_drug_likeness_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    molecular_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    ring_count = Descriptors.RingCount(mol)
    heavy_atom_count = Descriptors.HeavyAtomCount(mol)

    # Lipinski Rule of 5
    lipinski_rules = {
        "Molecular Weight <= 500": molecular_weight <= 500,
        "LogP <= 5": logp <= 5,
        "H-Bond Donors <= 5": h_donors <= 5,
        "H-Bond Acceptors <= 10": h_acceptors <= 10
    }

    lipinski_violations = sum(
        1 for passed in lipinski_rules.values()
        if not passed
    )

    if lipinski_violations == 0:
        lipinski_status = "Pass"
        drug_likeness_label = "Good Drug-Likeness"
    elif lipinski_violations == 1:
        lipinski_status = "Acceptable"
        drug_likeness_label = "Moderate Drug-Likeness"
    else:
        lipinski_status = "Fail"
        drug_likeness_label = "Poor Drug-Likeness"

    # Veber Rule
    veber_rules = {
        "Rotatable Bonds <= 10": rotatable_bonds <= 10,
        "TPSA <= 140": tpsa <= 140
    }

    veber_violations = sum(
        1 for passed in veber_rules.values()
        if not passed
    )

    veber_status = "Pass" if veber_violations == 0 else "Fail"

    # Ghose Filter
    ghose_rules = {
        "160 <= Molecular Weight <= 480": 160 <= molecular_weight <= 480,
        "-0.4 <= LogP <= 5.6": -0.4 <= logp <= 5.6,
        "Atom Count 20 to 70": 20 <= heavy_atom_count <= 70
    }

    ghose_violations = sum(
        1 for passed in ghose_rules.values()
        if not passed
    )

    ghose_status = "Pass" if ghose_violations == 0 else "Fail"

    # Lead-likeness
    lead_rules = {
        "Molecular Weight <= 350": molecular_weight <= 350,
        "LogP <= 3.5": logp <= 3.5,
        "Rotatable Bonds <= 7": rotatable_bonds <= 7
    }

    lead_violations = sum(
        1 for passed in lead_rules.values()
        if not passed
    )

    if lead_violations == 0:
        lead_status = "Lead-like"
    elif lead_violations == 1:
        lead_status = "Moderately Lead-like"
    else:
        lead_status = "Not Lead-like"

    # PAINS alerts using RDKit FilterCatalog
    pains_alert_count = 0
    pains_alerts = []

    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)

        catalog = FilterCatalog(params)
        matches = catalog.GetMatches(mol)

        pains_alert_count = len(matches)
        pains_alerts = [
            str(match.GetDescription())
            for match in matches
        ]

    except Exception:
        pains_alert_count = -1
        pains_alerts = ["PAINS screening unavailable"]

    if pains_alert_count == 0:
        pains_status = "No PAINS Alerts"
    elif pains_alert_count == -1:
        pains_status = "PAINS Screening Unavailable"
    else:
        pains_status = "PAINS Alert(s) Found"

    # Approximate synthetic accessibility proxy
    # Lower is better. This is NOT the official SA score, but a transparent RDKit-based estimate.
    complexity_penalty = (
        0.015 * molecular_weight
        +
        0.35 * ring_count
        +
        0.25 * rotatable_bonds
        +
        0.10 * h_acceptors
        +
        0.15 * h_donors
    )

    synthetic_accessibility_estimate = round(
        min(10, max(1, complexity_penalty / 3)),
        2
    )

    if synthetic_accessibility_estimate <= 3:
        sa_status = "Likely Easy"
    elif synthetic_accessibility_estimate <= 6:
        sa_status = "Moderate"
    else:
        sa_status = "Potentially Difficult"

    if tpsa <= 140:
        tpsa_status = "Acceptable"
    else:
        tpsa_status = "High TPSA / Lower Permeability Risk"

    if rotatable_bonds <= 10:
        flexibility_status = "Acceptable Flexibility"
    else:
        flexibility_status = "High Flexibility"

    bioavailability_score = 0

    if molecular_weight <= 500:
        bioavailability_score += 1

    if logp <= 5:
        bioavailability_score += 1

    if h_donors <= 5:
        bioavailability_score += 1

    if h_acceptors <= 10:
        bioavailability_score += 1

    if tpsa <= 140:
        bioavailability_score += 1

    if rotatable_bonds <= 10:
        bioavailability_score += 1

    if pains_alert_count == 0:
        bioavailability_score += 1

    bioavailability_percent = round(
        bioavailability_score / 7 * 100,
        2
    )

    if bioavailability_percent >= 80:
        bioavailability_label = "High Bioavailability Potential"
    elif bioavailability_percent >= 50:
        bioavailability_label = "Moderate Bioavailability Potential"
    else:
        bioavailability_label = "Low Bioavailability Potential"

    return {
        "Molecular_Weight": round(molecular_weight, 2),
        "LogP": round(logp, 2),
        "H_Bond_Donors": int(h_donors),
        "H_Bond_Acceptors": int(h_acceptors),
        "TPSA": round(tpsa, 2),
        "Rotatable_Bonds": int(rotatable_bonds),
        "Ring_Count": int(ring_count),
        "Heavy_Atom_Count": int(heavy_atom_count),

        "Lipinski_Violations": int(lipinski_violations),
        "Lipinski_Status": lipinski_status,
        "Drug_Likeness_Label": drug_likeness_label,

        "Veber_Violations": int(veber_violations),
        "Veber_Status": veber_status,

        "Ghose_Violations": int(ghose_violations),
        "Ghose_Status": ghose_status,

        "Lead_Likeness_Violations": int(lead_violations),
        "Lead_Likeness_Status": lead_status,

        "PAINS_Alert_Count": int(pains_alert_count),
        "PAINS_Status": pains_status,
        "PAINS_Alerts": "; ".join(pains_alerts),

        "Synthetic_Accessibility_Estimate": synthetic_accessibility_estimate,
        "Synthetic_Accessibility_Status": sa_status,

        "TPSA_Status": tpsa_status,
        "Flexibility_Status": flexibility_status,
        "Bioavailability_Score_%": bioavailability_percent,
        "Bioavailability_Label": bioavailability_label,

        "Rule_MW_<=500": lipinski_rules["Molecular Weight <= 500"],
        "Rule_LogP_<=5": lipinski_rules["LogP <= 5"],
        "Rule_HBD_<=5": lipinski_rules["H-Bond Donors <= 5"],
        "Rule_HBA_<=10": lipinski_rules["H-Bond Acceptors <= 10"],

        "Rule_Veber_RB_<=10": veber_rules["Rotatable Bonds <= 10"],
        "Rule_Veber_TPSA_<=140": veber_rules["TPSA <= 140"],

        "Rule_Ghose_MW_160_480": ghose_rules["160 <= Molecular Weight <= 480"],
        "Rule_Ghose_LogP_-0.4_5.6": ghose_rules["-0.4 <= LogP <= 5.6"],
        "Rule_Ghose_Atom_Count_20_70": ghose_rules["Atom Count 20 to 70"],

        "Rule_Lead_MW_<=350": lead_rules["Molecular Weight <= 350"],
        "Rule_Lead_LogP_<=3.5": lead_rules["LogP <= 3.5"],
        "Rule_Lead_RB_<=7": lead_rules["Rotatable Bonds <= 7"]
    }


def generate_drug_likeness_report_dataframe(
    molecule_name,
    smiles
):
    properties = calculate_drug_likeness_properties(
        smiles
    )

    if properties is None:
        return pd.DataFrame()

    report_data = {
        "Molecule_Name": molecule_name,
        "SMILES": smiles
    }

    report_data.update(properties)

    return pd.DataFrame([report_data])


def generate_batch_drug_likeness_dataframe(input_df):

    results = []

    for _, row in input_df.iterrows():

        smiles = str(row.get("SMILES", "")).strip()

        if not smiles:
            continue

        molecule_name = str(
            row.get("Molecule_Name", row.get("Name", "Unknown Molecule"))
        )

        report_df = generate_drug_likeness_report_dataframe(
            molecule_name=molecule_name,
            smiles=smiles
        )

        if not report_df.empty:
            results.append(report_df.iloc[0].to_dict())
        else:
            results.append({
                "Molecule_Name": molecule_name,
                "SMILES": smiles,
                "Error": "Invalid SMILES"
            })

    return pd.DataFrame(results)


def create_drug_likeness_pdf_report(
    molecule_name,
    smiles,
    report_df
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "Drug-Likeness Analysis Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"<b>Molecule:</b> {molecule_name}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>SMILES:</b> {smiles}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 12))

    if report_df.empty:
        story.append(
            Paragraph(
                "No drug-likeness results available.",
                styles["Normal"]
            )
        )
    else:
        row = report_df.iloc[0].to_dict()

        table_data = [["Property", "Value"]]

        selected_keys = [
            "Molecular_Weight",
            "LogP",
            "H_Bond_Donors",
            "H_Bond_Acceptors",
            "TPSA",
            "Rotatable_Bonds",
            "Lipinski_Status",
            "Veber_Status",
            "Ghose_Status",
            "Lead_Likeness_Status",
            "PAINS_Status",
            "Synthetic_Accessibility_Estimate",
            "Synthetic_Accessibility_Status",
            "Bioavailability_Score_%",
            "Bioavailability_Label"
        ]

        for key in selected_keys:
            if key in row:
                table_data.append([key, str(row[key])])

        report_table = Table(
            table_data,
            colWidths=[220, 250]
        )

        report_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(report_table)

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Note: Drug-likeness rules are heuristic screening tools and should not be interpreted as clinical or regulatory conclusions.",
            styles["Italic"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()

def pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        40,
        25,
        f"Hybrid GNN AI Cheminformatics Platform | Page {doc.page}"
    )
    canvas.restoreState()


def create_prediction_pdf_report(
    molecule_name,
    smiles,
    model_used,
    prediction_k,
    prediction_c,
    confidence,
    uncertainty_range,
    confidence_label,
    model_difference,
    properties_df,
    molecule_image,
    model_comparison_df=None,
    shap_df=None,
    similar_df=None
):
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    footer_note_style = ParagraphStyle(
        "FooterNote",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=colors.grey
    )

    story = []

    story.append(Paragraph("AI-Based Melting Point Prediction Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Hybrid GNN AI Cheminformatics Platform", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    molecule_data = [
        ["Field", "Value"],
        ["Molecule Name / IUPAC", molecule_name],
        ["SMILES", smiles],
        ["Model Used", model_used],
        ["Predicted Melting Point (K)", f"{prediction_k:.2f} K"],
        ["Predicted Melting Point (°C)", f"{prediction_c:.2f} °C"],
        ["Confidence", f"{confidence}%"],
        ["Confidence Category", confidence_label],
        ["Model Difference", f"{model_difference:.2f} K"],
        ["Estimated Uncertainty", f"± {uncertainty_range:.2f} K"]
    ]

    molecule_table = Table(
        molecule_data,
        colWidths=[180, 300]
    )

    molecule_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(molecule_table)
    story.append(Spacer(1, 16))

    if molecule_image is not None:
        img_buffer = BytesIO()
        molecule_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        story.append(Paragraph("Molecular Structure", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(Image(img_buffer, width=220, height=220))
        story.append(Spacer(1, 16))

    if model_comparison_df is not None and not model_comparison_df.empty:
        story.append(Paragraph("Model Comparison Table", styles["Heading2"]))
        story.append(Spacer(1, 8))

        model_data = [list(model_comparison_df.columns)]

        for _, row in model_comparison_df.iterrows():
            model_data.append([str(row[col]) for col in model_comparison_df.columns])

        model_table = Table(
            model_data,
            colWidths=[180, 140, 140]
        )

        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(model_table)
        story.append(Spacer(1, 16))

    story.append(Paragraph("RDKit Molecular Properties", styles["Heading2"]))
    story.append(Spacer(1, 8))

    prop_data = [["Property", "Value"]]

    for _, row in properties_df.iterrows():
        prop_data.append([str(row["Property"]), str(row["Value"])])

    prop_table = Table(
        prop_data,
        colWidths=[220, 220]
    )

    prop_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(prop_table)
    story.append(Spacer(1, 16))

    if shap_df is not None and not shap_df.empty:
        story.append(Paragraph("Top SHAP Features", styles["Heading2"]))
        story.append(Spacer(1, 8))

        shap_display_df = clean_shap_dataframe(
            shap_df,
            top_n=10
        )

        shap_data = [list(shap_display_df.columns)]

        for _, row in shap_display_df.iterrows():
            shap_data.append([str(row[col]) for col in shap_display_df.columns])

        shap_table = Table(
            shap_data,
            colWidths=[150 for _ in shap_display_df.columns]
        )

        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(shap_table)
        story.append(Spacer(1, 16))

    if similar_df is not None and not similar_df.empty:
        story.append(Paragraph("Top 10 Similar Molecules", styles["Heading2"]))
        story.append(Spacer(1, 8))

        similar_data = [list(similar_df.columns)]

        for _, row in similar_df.iterrows():
            similar_data.append([str(row[col]) for col in similar_df.columns])

        similar_table = Table(
            similar_data,
            colWidths=[180, 220, 90]
        )

        similar_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(similar_table)
        story.append(Spacer(1, 16))

    story.append(Paragraph("Interpretation Note", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Prediction confidence is estimated using agreement between RDKit LightGBM "
            "and Hybrid Descriptor + GAT models. Smaller disagreement indicates higher confidence.",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Professional Footer: This report was generated automatically by the Hybrid GNN AI "
            "Cheminformatics Platform for research, academic, and portfolio demonstration purposes. "
            "Predictions should be experimentally validated before scientific or industrial use.",
            footer_note_style
        )
    )

    doc.build(
        story,
        onFirstPage=pdf_footer,
        onLaterPages=pdf_footer
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def create_batch_summary_pdf(batch_df):
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Batch Melting Point Prediction Summary", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Hybrid GNN AI Cheminformatics Platform", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["BodyText"]
        )
    )
    story.append(Spacer(1, 12))

    total = len(batch_df)
    avg_pred = batch_df["Ensemble_Prediction_K"].mean()
    avg_conf = batch_df["Confidence_%"].mean()
    avg_uncertainty = batch_df["Estimated_Uncertainty_K"].mean()
    high_count = len(batch_df[batch_df["Confidence_Label"] == "High Confidence"])
    moderate_count = len(batch_df[batch_df["Confidence_Label"] == "Moderate Confidence"])
    low_count = len(batch_df[batch_df["Confidence_Label"] == "Low Confidence"])

    summary_data = [
        ["Metric", "Value"],
        ["Total Successful Molecules", total],
        ["Average Ensemble Prediction (K)", round(avg_pred, 2)],
        ["Average Confidence (%)", round(avg_conf, 2)],
        ["Average Estimated Uncertainty (K)", round(avg_uncertainty, 2)],
        ["High Confidence Count", high_count],
        ["Moderate Confidence Count", moderate_count],
        ["Low Confidence Count", low_count]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[260, 180]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Top Batch Prediction Results", styles["Heading2"]))
    story.append(Spacer(1, 8))

    display_cols = [
        "SMILES",
        "RDKit_LightGBM_K",
        "Hybrid_GAT_K",
        "Ensemble_Prediction_K",
        "Ensemble_Prediction_C",
        "Confidence_%",
        "Confidence_Label"
    ]

    display_df = batch_df[display_cols].head(25).copy()
    table_data = [display_df.columns.tolist()]

    for _, row in display_df.iterrows():
        table_data.append([str(row[col]) for col in display_df.columns])

    result_table = Table(table_data, repeatRows=1)
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(result_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Interpretation Note", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Batch confidence is estimated from agreement between RDKit LightGBM "
            "and Hybrid Descriptor + GAT predictions. Ensemble prediction uses "
            "40% RDKit LightGBM and 60% Hybrid GAT.",
            styles["BodyText"]
        )
    )

    doc.build(
        story,
        onFirstPage=pdf_footer,
        onLaterPages=pdf_footer
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ==================================================
# MAIN APP
# ==================================================

if st.session_state["authentication_status"] is True:

    create_prediction_table()

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome {st.session_state.get('name', 'User')}")

    st.title("🧪 Hybrid GNN AI Cheminformatics Platform")



    # Full molecule dataset check


    try:


        _dataset_check_df = load_molecule_dataset()


        _dataset_source = (


            _dataset_check_df["Dataset_Source_File"].iloc[0]


            if "Dataset_Source_File" in _dataset_check_df.columns


            else "Unknown"


        )



        if len(_dataset_check_df) < 500:


            st.warning(


                f"Dataset loaded: {len(_dataset_check_df)} molecules from {_dataset_source}. "


                "This looks like a small/test dataset. Please confirm all_smiles_with_names.csv is present."


            )


        else:


            st.success(


                f"Full molecule dataset loaded: {len(_dataset_check_df)} molecules from {_dataset_source}"


            )



    except Exception as dataset_error:


        st.error(f"Dataset check failed: {dataset_error}")

    st.write(
        "Predict molecular melting point using RDKit descriptors, LightGBM, Hybrid GAT AI, "
        "Ensemble AI, molecule search, similarity search, PNG export, 3D visualization, "
        "uncertainty estimation, enhanced PDF reporting, batch reports, dashboard analytics, "
        "and Murcko scaffold analysis."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "Single Molecule Prediction",
        "Molecule Explorer",
        "Batch CSV Prediction",
        "Use Saved Full Dataset",
        "Prediction History",
        "Dashboard Summary",
        "Scaffold Analysis",
        "OOD Detection",
        "Chemical Space PCA",
        "Chemical Space t-SNE",
        "Chemical Space UMAP",
        "Interactive Plotly UMAP + AI Overlay + Drug-Likeness Analysis",
        "Drug-Likeness Analysis"
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
        selected_name = "Not Selected"

        if input_mode == "Select from Dataset":

            try:
                smiles_df = load_molecule_dataset()

                st.info(
                    "Not sure which molecule to enter? Search or browse the "
                    "available molecule catalog below, then select a molecule "
                    "from the filtered dropdown."
                )

                catalog_csv = smiles_df[
                    [
                        "Molecule_Name",
                        "SMILES"
                    ]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Full Molecule Catalog CSV",
                    data=catalog_csv,
                    file_name="available_molecule_catalog.csv",
                    mime="text/csv",
                    key="download_single_molecule_catalog"
                )

                with st.expander("View Available Molecule Catalog", expanded=True):

                    if "catalog_search_reset_counter" not in st.session_state:
                        st.session_state["catalog_search_reset_counter"] = 0

                    search_key = (
                        "single_molecule_catalog_search_"
                        + str(st.session_state["catalog_search_reset_counter"])
                    )

                    selectbox_key = (
                        "single_selectbox_filtered_catalog_"
                        + str(st.session_state["catalog_search_reset_counter"])
                    )

                    col_search, col_reset = st.columns([5, 1])

                    with col_search:

                        molecule_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=search_key,
                            placeholder="Example: ethanol, benz, acid, CCO"
                        )

                    with col_reset:

                        st.write("")

                        st.write("")

                        if st.button(
                            "Clear / Reset",
                            key="clear_single_molecule_catalog_search"
                        ):

                            for key_to_clear in [
                                "selected_catalog_molecule_name",
                                "selected_catalog_smiles"
                            ]:

                                if key_to_clear in st.session_state:
                                    del st.session_state[key_to_clear]

                            st.session_state["catalog_search_reset_counter"] += 1

                            st.rerun()

                    if molecule_search_query.strip() != "":

                        filtered_smiles_df = smiles_df[
                            smiles_df["Molecule_Name"].str.contains(
                                molecule_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                            |
                            smiles_df["SMILES"].str.contains(
                                molecule_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                        ].copy()

                    else:

                        filtered_smiles_df = smiles_df.copy()

                    st.info(
                        f"Matching molecules found: {len(filtered_smiles_df)} "
                        f"out of {len(smiles_df)}"
                    )

                    display_paginated_molecule_table(
                        df=filtered_smiles_df,
                        table_key="filtered_smiles_df_paginated_catalog",
                        rows_per_page=100,
                        columns=[
                            "Molecule_Name",
                            "SMILES"
                        ]
                    )



                    filtered_catalog_csv = filtered_smiles_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Filtered Molecule List CSV",
                        data=filtered_catalog_csv,
                        file_name="filtered_molecule_catalog.csv",
                        mime="text/csv",
                        key="download_filtered_molecule_catalog"
                    )

                    st.markdown("---")

                    if filtered_smiles_df.empty:

                        st.warning(
                            "No molecule found for this search. Please try another name or SMILES."
                        )

                    else:

                        st.subheader("Select Molecule from Search Results")

                        selected_index = st.selectbox(
                            "Choose molecule from filtered list",
                            options=filtered_smiles_df.index,
                            format_func=lambda x: filtered_smiles_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=selectbox_key
                        )

                        selected_name = filtered_smiles_df.loc[
                            selected_index,
                            "Molecule_Name"
                        ]

                        manual_smiles = filtered_smiles_df.loc[
                            selected_index,
                            "SMILES"
                        ]

                        # Every dropdown selection immediately becomes
                        # the active prediction molecule.
                        st.session_state["selected_catalog_molecule_name"] = (
                            selected_name
                        )

                        st.session_state["selected_catalog_smiles"] = (
                            manual_smiles
                        )

                        st.success(
                            "Selected molecule is now added to the prediction input."
                        )

                        st.caption(
                            "Note: Standard Streamlit tables do not support reliable double-click row selection. "
                            "Use the dropdown above after searching; it updates the prediction molecule automatically."
                        )

                if filtered_smiles_df.empty:

                    manual_smiles = st.text_input(
                        "Enter SMILES manually",
                        value="CCO",
                        key="fallback_single_smiles_after_empty_filter"
                    )

                    selected_name = "Manual Input"

                else:

                    selected_name = st.session_state.get(
                        "selected_catalog_molecule_name",
                        filtered_smiles_df.iloc[0]["Molecule_Name"]
                    )

                    manual_smiles = st.session_state.get(
                        "selected_catalog_smiles",
                        filtered_smiles_df.iloc[0]["SMILES"]
                    )

                    st.subheader("Current Molecule Used for Prediction")

                    st.success(f"Selected Molecule Name: {selected_name}")
                    st.success(f"Selected SMILES: {manual_smiles}")

                    st.info(
                        "This selected molecule is now automatically used by the prediction model below."
                    )

                    st.subheader("Copy Selected Molecule Details")

                    safe_copy_name_key = make_safe_filename(
                        selected_name
                    )

                    safe_copy_smiles_key = make_safe_filename(
                        manual_smiles
                    )

                    st.text_area(
                        "Copy Molecule Name",
                        value=selected_name,
                        height=70,
                        key=f"copy_name_{safe_copy_name_key}"
                    )

                    st.text_area(
                        "Copy SMILES",
                        value=manual_smiles,
                        height=70,
                        key=f"copy_smiles_{safe_copy_smiles_key}"
                    )

                    st.code(
                        f"Molecule Name: {selected_name}\n"
                        f"SMILES: {manual_smiles}",
                        language="text"
                    )

            except Exception as e:

                st.error(f"Dataset loading failed: {e}")

                manual_smiles = st.text_input(
                    "Enter SMILES",
                    value="CCO",
                    key="fallback_single_smiles"
                )

        elif input_mode == "Enter Custom SMILES":

            manual_smiles = st.text_input(
                "Enter Custom SMILES",
                value="CCO",
                key="custom_smiles"
            )

            selected_name = "Custom SMILES Input"

        else:

            compound_name = st.text_input(
                "Enter Molecule Name",
                value="ethanol",
                key="compound_name_input"
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

            manual_smiles = st.session_state.get("converted_smiles", "CCO")
            selected_name = st.session_state.get("compound_name", compound_name)

            st.info(f"Current Molecule Name: {selected_name}")
            st.info(f"Current SMILES: {manual_smiles}")

        model_choice = st.radio(
            "Select Prediction Model",
            [
                "RDKit LightGBM",
                "Hybrid Descriptor + GAT",
                "Ensemble AI Prediction"
            ]
        )

        mol = Chem.MolFromSmiles(manual_smiles)

        molecule_image = None
        properties_df = None

        if mol is not None:

            molecule_image = Draw.MolToImage(mol, size=(400, 400))
            st.image(molecule_image, caption="Molecular Structure")

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
            properties_df["Value"] = properties_df["Value"].astype(str)
            st.dataframe(properties_df)

        else:
            st.error("Invalid SMILES string.")

        if st.button("Predict Melting Point"):

            try:

                rdkit_prediction = float(predict_melting_point(manual_smiles))
                hybrid_prediction = float(predict_hybrid_gat(manual_smiles))

                model_comparison_df = pd.DataFrame({
                    "Model": [
                        "RDKit LightGBM",
                        "Hybrid Descriptor + GAT"
                    ],
                    "Prediction_K": [
                        round(rdkit_prediction, 2),
                        round(hybrid_prediction, 2)
                    ]
                })

                if model_choice == "RDKit LightGBM":
                    prediction_k = rdkit_prediction

                elif model_choice == "Hybrid Descriptor + GAT":
                    prediction_k = hybrid_prediction

                else:
                    prediction_k = (
                        0.4 * rdkit_prediction
                        +
                        0.6 * hybrid_prediction
                    )

                    model_comparison_df = pd.concat([
                        model_comparison_df,
                        pd.DataFrame({
                            "Model": ["Final Ensemble"],
                            "Prediction_K": [round(prediction_k, 2)]
                        })
                    ], ignore_index=True)

                    st.subheader("Ensemble Model Details")

                    display_comparison_df = model_comparison_df.copy()
                    display_comparison_df["Prediction_C"] = (
                        display_comparison_df["Prediction_K"] - 273.15
                    ).round(2)

                    st.dataframe(display_comparison_df)

                    st.success(
                        "Final Ensemble Prediction calculated using "
                        "40% RDKit LightGBM + 60% Hybrid GAT."
                    )

                model_comparison_df["Prediction_C"] = (
                    model_comparison_df["Prediction_K"] - 273.15
                ).round(2)

                uncertainty_results = calculate_prediction_uncertainty(
                    rdkit_prediction,
                    hybrid_prediction
                )

                st.subheader("Prediction Confidence & Uncertainty")

                col_u1, col_u2, col_u3 = st.columns(3)

                with col_u1:
                    st.metric("Confidence %", f"{uncertainty_results['confidence']}%")

                with col_u2:
                    st.metric("Model Difference", f"{uncertainty_results['difference']:.2f} K")

                with col_u3:
                    st.metric(
                        "Estimated ± Uncertainty",
                        f"± {uncertainty_results['uncertainty_range']:.2f} K"
                    )

                if uncertainty_results["confidence"] >= 85:
                    st.success(uncertainty_results["confidence_label"])
                elif uncertainty_results["confidence"] >= 70:
                    st.warning(uncertainty_results["confidence_label"])
                else:
                    st.error(uncertainty_results["confidence_label"])

                prediction_c = prediction_k - 273.15

                st.success("Prediction completed successfully")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Melting Point (Kelvin)", f"{prediction_k:.2f} K")

                with col2:
                    st.metric("Melting Point (Celsius)", f"{prediction_c:.2f} °C")

                st.info(f"Molecule Name: {selected_name}")
                st.info(f"SMILES: {manual_smiles}")
                st.info(f"Model Used: {model_choice}")

                log_prediction(
                    username=st.session_state.get("username", "unknown"),
                    smiles=manual_smiles,
                    model_used=model_choice,
                    prediction_k=prediction_k,
                    prediction_c=prediction_c,
                    status=(
                        "Success | "
                        f"Confidence: {uncertainty_results['confidence']}% | "
                        f"Uncertainty: ±{uncertainty_results['uncertainty_range']} K"
                    )
                )

                pdf_shap_df = pd.DataFrame()

                if model_choice == "RDKit LightGBM":

                    st.subheader("RDKit LightGBM SHAP Explanation")

                    pdf_shap_df = explain_prediction(manual_smiles)
                    st.dataframe(pdf_shap_df)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(pdf_shap_df["Feature"], pdf_shap_df["SHAP_Value"])
                    ax.set_xlabel("SHAP Value")
                    ax.set_ylabel("Feature")
                    ax.set_title("Top RDKit LightGBM SHAP Contributions")
                    st.pyplot(fig)

                elif model_choice == "Hybrid Descriptor + GAT":

                    st.subheader("Hybrid GAT SHAP Explanation")

                    pdf_shap_df = explain_hybrid_gat_prediction(
                        manual_smiles,
                        top_n=10
                    )

                    st.dataframe(pdf_shap_df)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(pdf_shap_df["Feature"], pdf_shap_df["SHAP_Value"])
                    ax.set_xlabel("SHAP Value")
                    ax.set_ylabel("Feature")
                    ax.set_title("Top Hybrid GAT SHAP Contributions")
                    st.pyplot(fig)

                    st.subheader("Hybrid GAT Feature Importance")

                    hybrid_importance_df = get_hybrid_feature_importance(top_n=15)
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

                else:
                    st.info(
                        "For Ensemble AI Prediction, the report includes "
                        "model comparison, uncertainty, molecular properties, "
                        "and similar molecules. Use individual models for SHAP-specific PDF sections."
                    )

                full_similarity_df = load_molecule_dataset()

                pdf_similar_df = find_top_similar_molecules(
                    query_smiles=manual_smiles,
                    molecule_df=full_similarity_df,
                    top_n=10
                )

                if properties_df is not None and molecule_image is not None:

                    pdf_bytes = create_prediction_pdf_report(
                        molecule_name=selected_name,
                        smiles=manual_smiles,
                        model_used=model_choice,
                        prediction_k=prediction_k,
                        prediction_c=prediction_c,
                        confidence=uncertainty_results["confidence"],
                        uncertainty_range=uncertainty_results["uncertainty_range"],
                        confidence_label=uncertainty_results["confidence_label"],
                        model_difference=uncertainty_results["difference"],
                        properties_df=properties_df,
                        molecule_image=molecule_image,
                        model_comparison_df=model_comparison_df,
                        shap_df=pdf_shap_df,
                        similar_df=pdf_similar_df
                    )

                    safe_pdf_name = make_safe_filename(selected_name)

                    st.download_button(
                        label="Download Enhanced Prediction Report PDF",
                        data=pdf_bytes,
                        file_name=f"{safe_pdf_name}_enhanced_prediction_report.pdf",
                        mime="application/pdf"
                    )

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

        st.subheader("Molecule Explorer")

        st.write(
            "Search, browse, copy, visualize, and analyze molecules from the available dataset."
        )

        try:
            explorer_full_df = load_molecule_dataset()

            explorer_catalog_csv = explorer_full_df[
                [
                    "Molecule_Name",
                    "SMILES"
                ]
            ].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Full Molecule Catalog CSV",
                data=explorer_catalog_csv,
                file_name="available_molecule_catalog.csv",
                mime="text/csv",
                key="download_explorer_full_catalog"
            )

            if "explorer_search_reset_counter" not in st.session_state:
                st.session_state["explorer_search_reset_counter"] = 0

            explorer_search_key = (
                "explorer_search_input_"
                + str(st.session_state["explorer_search_reset_counter"])
            )

            explorer_selectbox_key = (
                "explorer_selectbox_"
                + str(st.session_state["explorer_search_reset_counter"])
            )

            col_exp_search, col_exp_reset = st.columns([5, 1])

            with col_exp_search:

                explorer_search_query = st.text_input(
                    "Search molecule by IUPAC/name or SMILES",
                    value="",
                    key=explorer_search_key,
                    placeholder="Example: ethanol, benz, acid, CCO"
                )

            with col_exp_reset:

                st.write("")
                st.write("")

                if st.button(
                    "Clear / Reset",
                    key="clear_explorer_search"
                ):

                    for key_to_clear in [
                        "explorer_selected_molecule_name",
                        "explorer_selected_smiles"
                    ]:
                        if key_to_clear in st.session_state:
                            del st.session_state[key_to_clear]

                    st.session_state["explorer_search_reset_counter"] += 1
                    st.rerun()

            if explorer_search_query.strip() != "":

                explorer_df = explorer_full_df[
                    explorer_full_df["Molecule_Name"].str.contains(
                        explorer_search_query,
                        case=False,
                        na=False,
                        regex=False
                    )
                    |
                    explorer_full_df["SMILES"].str.contains(
                        explorer_search_query,
                        case=False,
                        na=False,
                        regex=False
                    )
                ].copy()

            else:

                explorer_df = explorer_full_df.copy()

            st.info(
                f"Matching molecules found: {len(explorer_df)} "
                f"out of {len(explorer_full_df)}"
            )

            display_paginated_molecule_table(
                df=explorer_df,
                table_key="explorer_df_paginated_catalog",
                rows_per_page=100,
                columns=[
                    "Molecule_Name",
                    "SMILES"
                ]
            )



            filtered_explorer_csv = explorer_df[
                [
                    "Molecule_Name",
                    "SMILES"
                ]
            ].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Filtered Molecule List CSV",
                data=filtered_explorer_csv,
                file_name="filtered_explorer_molecule_catalog.csv",
                mime="text/csv",
                key="download_explorer_filtered_catalog"
            )

            if explorer_df.empty:

                st.warning(
                    "No molecule found for this search. Please try another name or SMILES."
                )

            else:

                st.subheader("Select Molecule from Search Results")

                explorer_selected_index = st.selectbox(
                    "Choose molecule from filtered list",
                    options=explorer_df.index,
                    format_func=lambda x: explorer_df.loc[
                        x,
                        "Molecule_Display"
                    ],
                    key=explorer_selectbox_key
                )

                explorer_selected_name = explorer_df.loc[
                    explorer_selected_index,
                    "Molecule_Name"
                ]

                explorer_selected_smiles = explorer_df.loc[
                    explorer_selected_index,
                    "SMILES"
                ]

                st.session_state["explorer_selected_molecule_name"] = (
                    explorer_selected_name
                )

                st.session_state["explorer_selected_smiles"] = (
                    explorer_selected_smiles
                )

                safe_name = make_safe_filename(explorer_selected_name)

                st.subheader("Current Molecule Selected in Explorer")

                st.success(f"Selected Molecule Name: {explorer_selected_name}")
                st.success(f"Selected SMILES: {explorer_selected_smiles}")

                st.info(
                    "This selected molecule is now used for structure visualization, "
                    "molecular properties, and similarity search below."
                )

                st.subheader("Copy Selected Molecule Details")

                safe_explorer_name_key = make_safe_filename(
                    explorer_selected_name
                )

                safe_explorer_smiles_key = make_safe_filename(
                    explorer_selected_smiles
                )

                st.text_area(
                    "Copy Molecule Name",
                    value=explorer_selected_name,
                    height=70,
                    key=f"explorer_copy_name_{safe_explorer_name_key}"
                )

                st.text_area(
                    "Copy SMILES",
                    value=explorer_selected_smiles,
                    height=70,
                    key=f"explorer_copy_smiles_{safe_explorer_smiles_key}"
                )

                st.code(
                    f"Molecule Name: {explorer_selected_name}\\n"
                        f"SMILES: {explorer_selected_smiles}",
                    language="text"
                )

                mol = Chem.MolFromSmiles(explorer_selected_smiles)

                if mol is not None:

                    st.subheader("2D Molecular Structure")

                    molecule_image = Draw.MolToImage(
                        mol,
                        size=(400, 400)
                    )

                    st.image(
                        molecule_image,
                        caption="2D Molecular Structure"
                    )

                    img_buffer = BytesIO()
                    molecule_image.save(
                        img_buffer,
                        format="PNG"
                    )

                    st.download_button(
                        label="Download Molecule Image PNG",
                        data=img_buffer.getvalue(),
                        file_name=f"{safe_name}_molecule.png",
                        mime="image/png"
                    )

                    st.subheader("3D Molecular Visualization")

                    with st.expander(
                        "Open Interactive 3D Molecule Viewer",
                        expanded=False
                    ):

                        st.info(
                            "If the viewer is blank, wait a few seconds or refresh the app. "
                            "Some molecules may fail 3D embedding depending on structure complexity."
                        )

                        show_3d_molecule(
                            explorer_selected_smiles,
                            width=650,
                            height=480,
                            viewer_key=f"explorer_3d_{safe_name}"
                        )

                    explorer_properties_df = pd.DataFrame({
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
                    st.dataframe(explorer_properties_df)

                    st.subheader("Top 10 Similar Molecules")

                    if st.button(
                        "Find Top 10 Similar Molecules",
                        key="explorer_find_top_10_similar"
                    ):

                        with st.spinner("Calculating molecular similarity..."):

                            full_similarity_df = load_molecule_dataset()

                            similar_df = find_top_similar_molecules(
                                query_smiles=explorer_selected_smiles,
                                molecule_df=full_similarity_df,
                                top_n=10
                            )

                        if similar_df.empty:

                            st.warning("No similar molecules found.")

                        else:

                            st.success("Top 10 similar molecules found.")
                            st.dataframe(similar_df)

                            similar_csv = similar_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download Similar Molecules CSV",
                                data=similar_csv,
                                file_name=f"{safe_name}_top_10_similar_molecules.csv",
                                mime="text/csv"
                            )

                else:

                    st.error("Invalid SMILES found in dataset.")

        except Exception as e:

            st.error(f"Molecule Explorer failed: {e}")

    with tab3:

        st.subheader("Batch CSV Prediction with Confidence + Ensemble + PDF Summary")

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

                if st.button("Run Enhanced Batch Prediction"):

                    batch_results = []
                    smiles_list = df["SMILES"].dropna().astype(str).tolist()

                    if len(smiles_list) == 0:
                        st.error("No valid SMILES found in uploaded CSV.")

                    else:
                        progress_bar = st.progress(0)

                        for i, smiles in enumerate(smiles_list):

                            try:
                                rdkit_pred = float(predict_melting_point(smiles))
                                hybrid_pred = float(predict_hybrid_gat(smiles))
                                ensemble_pred = 0.4 * rdkit_pred + 0.6 * hybrid_pred
                                uncertainty = calculate_prediction_uncertainty(
                                    rdkit_pred,
                                    hybrid_pred
                                )

                                batch_results.append({
                                    "SMILES": smiles,
                                    "RDKit_LightGBM_K": round(rdkit_pred, 2),
                                    "Hybrid_GAT_K": round(hybrid_pred, 2),
                                    "Ensemble_Prediction_K": round(ensemble_pred, 2),
                                    "Ensemble_Prediction_C": round(ensemble_pred - 273.15, 2),
                                    "Model_Difference_K": uncertainty["difference"],
                                    "Estimated_Uncertainty_K": uncertainty["uncertainty_range"],
                                    "Confidence_%": uncertainty["confidence"],
                                    "Confidence_Label": uncertainty["confidence_label"],
                                    "Status": "Success"
                                })

                            except Exception as e:
                                batch_results.append({
                                    "SMILES": smiles,
                                    "RDKit_LightGBM_K": None,
                                    "Hybrid_GAT_K": None,
                                    "Ensemble_Prediction_K": None,
                                    "Ensemble_Prediction_C": None,
                                    "Model_Difference_K": None,
                                    "Estimated_Uncertainty_K": None,
                                    "Confidence_%": None,
                                    "Confidence_Label": None,
                                    "Status": f"Failed: {e}"
                                })

                            progress_bar.progress(
                                int((i + 1) / len(smiles_list) * 100)
                            )

                        batch_df = pd.DataFrame(batch_results)

                        st.success("Enhanced batch prediction completed")
                        st.dataframe(batch_df)

                        csv = batch_df.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Download Enhanced Batch Predictions CSV",
                            data=csv,
                            file_name="enhanced_batch_predictions.csv",
                            mime="text/csv"
                        )

                        successful_batch_df = batch_df[
                            batch_df["Status"] == "Success"
                        ].copy()

                        if not successful_batch_df.empty:

                            pdf_bytes = create_batch_summary_pdf(successful_batch_df)

                            st.download_button(
                                "Download Batch Summary PDF",
                                data=pdf_bytes,
                                file_name="batch_prediction_summary_report.pdf",
                                mime="application/pdf"
                            )

                        else:
                            st.warning(
                                "No successful predictions available for PDF summary."
                            )

    with tab4:

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

    with tab5:

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

    with tab6:

        st.subheader("Dashboard Summary")

        dashboard_mode = st.radio(
            "Choose Dashboard Mode",
            [
                "Global Dashboard",
                "Current Molecule Dashboard"
            ],
            horizontal=True,
            key="dashboard_mode_selector"
        )

        if dashboard_mode == "Global Dashboard":

            st.write(
                "This dashboard summarizes all prediction activity stored in the local prediction history database."
            )

            rows = load_prediction_logs()

            if len(rows) == 0:
                st.info("No prediction history available for dashboard.")

            else:
                dashboard_df = pd.DataFrame(
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

                dashboard_df["Prediction K"] = pd.to_numeric(
                    dashboard_df["Prediction K"],
                    errors="coerce"
                )

                dashboard_df["Prediction °C"] = pd.to_numeric(
                    dashboard_df["Prediction °C"],
                    errors="coerce"
                )

                dashboard_df["Confidence %"] = dashboard_df["Status"].apply(
                    extract_confidence_from_status
                )

                dashboard_df["Prediction Result"] = dashboard_df["Status"].apply(
                    lambda x: "Success" if str(x).startswith("Success") else "Failed"
                )

                total_predictions = len(dashboard_df)
                success_count = len(
                    dashboard_df[dashboard_df["Prediction Result"] == "Success"]
                )
                failed_count = len(
                    dashboard_df[dashboard_df["Prediction Result"] == "Failed"]
                )

                avg_prediction_k = dashboard_df["Prediction K"].mean()
                avg_prediction_c = dashboard_df["Prediction °C"].mean()
                avg_confidence = dashboard_df["Confidence %"].mean()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Predictions", total_predictions)

                with col2:
                    st.metric("Successful", success_count)

                with col3:
                    st.metric("Failed", failed_count)

                with col4:
                    if pd.notna(avg_confidence):
                        st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                    else:
                        st.metric("Average Confidence", "N/A")

                col5, col6 = st.columns(2)

                with col5:
                    if pd.notna(avg_prediction_k):
                        st.metric(
                            "Average Melting Point (K)",
                            f"{avg_prediction_k:.2f} K"
                        )
                    else:
                        st.metric("Average Melting Point (K)", "N/A")

                with col6:
                    if pd.notna(avg_prediction_c):
                        st.metric(
                            "Average Melting Point (°C)",
                            f"{avg_prediction_c:.2f} °C"
                        )
                    else:
                        st.metric("Average Melting Point (°C)", "N/A")

                st.markdown("---")

                st.subheader("Success vs Failed Predictions")

                result_counts = dashboard_df["Prediction Result"].value_counts()

                fig_result, ax_result = plt.subplots(figsize=(6, 4))
                ax_result.bar(result_counts.index, result_counts.values)
                ax_result.set_xlabel("Prediction Result")
                ax_result.set_ylabel("Count")
                ax_result.set_title("Success vs Failed Predictions")

                st.pyplot(fig_result)

                st.subheader("Model Usage Count")

                model_counts = dashboard_df["Model Used"].value_counts()

                fig_model, ax_model = plt.subplots(figsize=(7, 4))
                ax_model.bar(model_counts.index, model_counts.values)
                ax_model.set_xlabel("Model")
                ax_model.set_ylabel("Count")
                ax_model.set_title("Model Usage Count")
                ax_model.tick_params(axis="x", rotation=20)

                st.pyplot(fig_model)

                st.subheader("Confidence Distribution")

                confidence_df = dashboard_df.dropna(subset=["Confidence %"])

                if confidence_df.empty:
                    st.info(
                        "No confidence values available yet. "
                        "Run predictions using the latest uncertainty-enabled version."
                    )

                else:
                    fig_conf, ax_conf = plt.subplots(figsize=(7, 4))
                    ax_conf.hist(confidence_df["Confidence %"], bins=10)
                    ax_conf.set_xlabel("Confidence %")
                    ax_conf.set_ylabel("Frequency")
                    ax_conf.set_title("Prediction Confidence Distribution")

                    st.pyplot(fig_conf)

                st.subheader("Recent Predictions")

                recent_df = dashboard_df.sort_values(
                    by="ID",
                    ascending=False
                ).head(10)

                st.dataframe(recent_df)

                dashboard_csv = dashboard_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Global Dashboard Data CSV",
                    data=dashboard_csv,
                    file_name="global_dashboard_summary_data.csv",
                    mime="text/csv"
                )

        else:

            st.write(
                "This dashboard analyzes one selected molecule using prediction, confidence, "
                "OOD status, scaffold, nearest molecule, similar molecules, and interpretation."
            )

            try:
                current_df = load_molecule_dataset()

                current_input_mode = st.radio(
                    "Choose Current Molecule Input Method",
                    [
                        "Select from Dataset",
                        "Enter Custom SMILES"
                    ],
                    horizontal=True,
                    key="current_dashboard_input_mode"
                )

                current_name = "Custom Input"
                current_smiles = "CCO"

                if current_input_mode == "Select from Dataset":

                    st.info(
                        "Search or browse the available molecule catalog below, "
                        "then select a molecule for the current molecule dashboard."
                    )

                    dashboard_catalog_csv = current_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Full Molecule Catalog CSV",
                        data=dashboard_catalog_csv,
                        file_name="dashboard_full_molecule_catalog.csv",
                        mime="text/csv",
                        key="download_dashboard_full_catalog"
                    )

                    if "dashboard_catalog_reset_counter" not in st.session_state:
                        st.session_state["dashboard_catalog_reset_counter"] = 0

                    dashboard_search_key = (
                        "current_dashboard_search_"
                        + str(st.session_state["dashboard_catalog_reset_counter"])
                    )

                    dashboard_selectbox_key = (
                        "current_dashboard_selectbox_"
                        + str(st.session_state["dashboard_catalog_reset_counter"])
                    )

                    with st.expander(
                        "View Available Molecule Catalog for Dashboard",
                        expanded=True
                    ):

                        col_dash_search, col_dash_reset = st.columns([5, 1])

                        with col_dash_search:

                            current_search_query = st.text_input(
                                "Search molecule by name or SMILES",
                                value="",
                                key=dashboard_search_key,
                                placeholder="Example: ethanol, benz, acid, CCO"
                            )

                        with col_dash_reset:

                            st.write("")
                            st.write("")

                            if st.button(
                                "Clear / Reset",
                                key="clear_current_dashboard_search"
                            ):

                                for key_to_clear in [
                                    "current_dashboard_selected_name",
                                    "current_dashboard_selected_smiles"
                                ]:
                                    if key_to_clear in st.session_state:
                                        del st.session_state[key_to_clear]

                                st.session_state[
                                    "dashboard_catalog_reset_counter"
                                ] += 1

                                st.rerun()

                        if current_search_query.strip() != "":

                            current_filtered_df = current_df[
                                current_df["Molecule_Name"].str.contains(
                                    current_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                                |
                                current_df["SMILES"].str.contains(
                                    current_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                            ].copy()

                        else:

                            current_filtered_df = current_df.copy()

                        st.info(
                            f"Matching molecules found: {len(current_filtered_df)} "
                            f"out of {len(current_df)}"
                        )

                        display_paginated_molecule_table(
                            df=current_filtered_df,
                            table_key="current_filtered_df_paginated_catalog",
                            rows_per_page=100,
                            columns=[
                                "Molecule_Name",
                                "SMILES"
                            ]
                        )



                        filtered_dashboard_csv = current_filtered_df[
                            [
                                "Molecule_Name",
                                "SMILES"
                            ]
                        ].to_csv(index=False).encode("utf-8")

                        st.download_button(
                            label="Download Filtered Dashboard Molecule List CSV",
                            data=filtered_dashboard_csv,
                            file_name="dashboard_filtered_molecule_catalog.csv",
                            mime="text/csv",
                            key="download_dashboard_filtered_catalog"
                        )

                        st.markdown("---")

                        if current_filtered_df.empty:

                            st.warning(
                                "No molecule found. Please try another search."
                            )

                        else:

                            st.subheader("Select Molecule for Current Dashboard")

                            current_selected_index = st.selectbox(
                                "Choose molecule from filtered list",
                                options=current_filtered_df.index,
                                format_func=lambda x: current_filtered_df.loc[
                                    x,
                                    "Molecule_Display"
                                ],
                                key=dashboard_selectbox_key
                            )

                            current_name = current_filtered_df.loc[
                                current_selected_index,
                                "Molecule_Name"
                            ]

                            current_smiles = current_filtered_df.loc[
                                current_selected_index,
                                "SMILES"
                            ]

                            st.session_state[
                                "current_dashboard_selected_name"
                            ] = current_name

                            st.session_state[
                                "current_dashboard_selected_smiles"
                            ] = current_smiles

                            st.success(
                                "Selected molecule is now added to the current molecule dashboard input."
                            )

                    if "current_filtered_df" in locals() and not current_filtered_df.empty:

                        current_name = st.session_state.get(
                            "current_dashboard_selected_name",
                            current_filtered_df.iloc[0]["Molecule_Name"]
                        )

                        current_smiles = st.session_state.get(
                            "current_dashboard_selected_smiles",
                            current_filtered_df.iloc[0]["SMILES"]
                        )

                    else:

                        current_name = "Dataset Selection"
                        current_smiles = "CCO"

                else:

                    current_name = "Custom Input"

                    current_smiles = st.text_input(
                        "Enter custom SMILES",
                        value="CCO",
                        key="current_dashboard_custom_smiles"
                    )

                st.subheader("Current Molecule Selected for Dashboard")

                st.success(f"Current Molecule: {current_name}")
                st.success(f"Current SMILES: {current_smiles}")

                st.text_area(
                    "Copy Current Molecule Name",
                    value=current_name,
                    height=70,
                    key=f"dashboard_copy_name_{make_safe_filename(current_name)}"
                )

                st.text_area(
                    "Copy Current SMILES",
                    value=current_smiles,
                    height=70,
                    key=f"dashboard_copy_smiles_{make_safe_filename(current_smiles)}"
                )

                if st.button(
                    "Generate Current Molecule Dashboard",
                    key="generate_current_molecule_dashboard"
                ):

                    mol = Chem.MolFromSmiles(current_smiles)

                    if mol is None:
                        st.error("Invalid SMILES. Please enter a valid molecule.")

                    else:
                        with st.spinner("Generating current molecule dashboard..."):

                            rdkit_pred = float(
                                predict_melting_point(current_smiles)
                            )

                            hybrid_pred = float(
                                predict_hybrid_gat(current_smiles)
                            )

                            ensemble_pred = (
                                0.4 * rdkit_pred
                                +
                                0.6 * hybrid_pred
                            )

                            uncertainty = calculate_prediction_uncertainty(
                                rdkit_pred,
                                hybrid_pred
                            )

                            ood_result = detect_ood_molecule(
                                current_smiles,
                                current_df
                            )

                            scaffold_smiles = get_murcko_scaffold(
                                current_smiles
                            )

                            similar_df = find_top_similar_molecules(
                                query_smiles=current_smiles,
                                molecule_df=current_df,
                                top_n=10
                            )

                        st.subheader("Current Molecule Prediction Summary")

                        col_a, col_b, col_c, col_d = st.columns(4)

                        with col_a:
                            st.metric(
                                "Ensemble Prediction",
                                f"{ensemble_pred:.2f} K"
                            )

                        with col_b:
                            st.metric(
                                "Prediction °C",
                                f"{ensemble_pred - 273.15:.2f} °C"
                            )

                        with col_c:
                            st.metric(
                                "Confidence",
                                f"{uncertainty['confidence']}%"
                            )

                        with col_d:
                            st.metric(
                                "OOD Status",
                                ood_result["OOD_Status"]
                            )

                        col_e, col_f, col_g = st.columns(3)

                        with col_e:
                            st.metric(
                                "RDKit LightGBM",
                                f"{rdkit_pred:.2f} K"
                            )

                        with col_f:
                            st.metric(
                                "Hybrid GAT",
                                f"{hybrid_pred:.2f} K"
                            )

                        with col_g:
                            st.metric(
                                "Model Difference",
                                f"{uncertainty['difference']:.2f} K"
                            )

                        if uncertainty["confidence_label"] == "High Confidence":
                            st.success(
                                f"Prediction Confidence: {uncertainty['confidence_label']}"
                            )
                        elif uncertainty["confidence_label"] == "Moderate Confidence":
                            st.warning(
                                f"Prediction Confidence: {uncertainty['confidence_label']}"
                            )
                        else:
                            st.error(
                                f"Prediction Confidence: {uncertainty['confidence_label']}"
                            )

                        if ood_result["OOD_Status"] == "In Distribution":
                            st.success(ood_result["Warning"])
                        elif ood_result["OOD_Status"] == "Borderline":
                            st.warning(ood_result["Warning"])
                        else:
                            st.error(ood_result["Warning"])

                        st.markdown("---")

                        st.subheader("Molecular Structure Comparison")

                        molecule_image = Draw.MolToImage(
                            mol,
                            size=(350, 350)
                        )

                        col_2d, col_3d = st.columns(2)

                        with col_2d:

                            st.markdown("### 2D Molecular Structure")

                            st.image(
                                molecule_image,
                                caption="Current Molecule 2D Structure"
                            )

                        with col_3d:

                            st.markdown("### 3D Molecular Structure")

                            st.info(
                                "Interactive 3D structure is generated automatically below."
                            )

                            show_3d_molecule(
                                current_smiles,
                                width=430,
                                height=400,
                                viewer_key="current_dashboard_3d"
                            )

                        st.markdown("---")

                        st.subheader("Molecular Properties")

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

                        properties_df["Value"] = properties_df["Value"].astype(str)
                        st.dataframe(properties_df)

                        st.markdown("---")

                        st.subheader("OOD and Nearest Molecule Details")

                        ood_display_df = pd.DataFrame([ood_result])
                        st.dataframe(ood_display_df)

                        nearest_smiles = ood_result["Nearest_SMILES"]
                        nearest_name = ood_result["Nearest_Molecule_Name"]

                        if nearest_smiles is not None:
                            nearest_mol = Chem.MolFromSmiles(nearest_smiles)

                            if nearest_mol is not None:
                                nearest_img = Draw.MolToImage(
                                    nearest_mol,
                                    size=(300, 300)
                                )

                                st.image(
                                    nearest_img,
                                    caption=f"Nearest Dataset Molecule: {nearest_name}"
                                )

                        st.markdown("---")

                        st.subheader("Murcko Scaffold")

                        st.code(
                            f"Murcko Scaffold: {scaffold_smiles}",
                            language="text"
                        )

                        if scaffold_smiles not in [None, "No Scaffold"]:
                            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

                            if scaffold_mol is not None:
                                scaffold_image = Draw.MolToImage(
                                    scaffold_mol,
                                    size=(300, 300)
                                )

                                st.image(
                                    scaffold_image,
                                    caption="Current Molecule Murcko Scaffold"
                                )

                        st.markdown("---")

                        st.subheader("Top 10 Similar Molecules")

                        if similar_df.empty:
                            st.warning("No similar molecules found.")
                        else:
                            st.dataframe(similar_df)

                        st.markdown("---")

                        st.subheader("Prediction Explanation")

                        explanation_text = (
                            f"The selected molecule ({current_name}) has an ensemble predicted "
                            f"melting point of {ensemble_pred:.2f} K "
                            f"({ensemble_pred - 273.15:.2f} °C). "
                            f"The RDKit LightGBM model predicted {rdkit_pred:.2f} K, while the "
                            f"Hybrid GAT model predicted {hybrid_pred:.2f} K. "
                            f"The model disagreement is {uncertainty['difference']:.2f} K, giving a "
                            f"confidence score of {uncertainty['confidence']}% and a confidence label of "
                            f"{uncertainty['confidence_label']}. "
                            f"The molecule is classified as {ood_result['OOD_Status']} based on nearest "
                            f"Tanimoto similarity to the known dataset. "
                            f"The nearest dataset molecule is {ood_result['Nearest_Molecule_Name']} "
                            f"with similarity {ood_result['Max_Tanimoto_Similarity']}."
                        )

                        st.info(explanation_text)

                        current_dashboard_df = pd.DataFrame([{
                            "Molecule_Name": current_name,
                            "SMILES": current_smiles,
                            "RDKit_LightGBM_K": round(rdkit_pred, 2),
                            "Hybrid_GAT_K": round(hybrid_pred, 2),
                            "Ensemble_Prediction_K": round(ensemble_pred, 2),
                            "Ensemble_Prediction_C": round(ensemble_pred - 273.15, 2),
                            "Confidence_%": uncertainty["confidence"],
                            "Confidence_Label": uncertainty["confidence_label"],
                            "Model_Difference_K": uncertainty["difference"],
                            "Estimated_Uncertainty_K": uncertainty["uncertainty_range"],
                            "OOD_Status": ood_result["OOD_Status"],
                            "Nearest_Molecule_Name": ood_result["Nearest_Molecule_Name"],
                            "Nearest_SMILES": ood_result["Nearest_SMILES"],
                            "Nearest_Similarity": ood_result["Max_Tanimoto_Similarity"],
                            "Murcko_Scaffold": scaffold_smiles
                        }])

                        current_csv = current_dashboard_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Current Molecule Dashboard CSV",
                            data=current_csv,
                            file_name="current_molecule_dashboard.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Current molecule dashboard failed: {e}")


    with tab7:

        st.subheader("Scaffold Analysis")

        st.write(
            "Analyze Bemis–Murcko scaffolds, explore molecular core structures, "
            "and identify molecules that share the same scaffold family."
        )

        try:

            molecule_df = load_molecule_dataset()

            scaffold_mode = st.radio(
                "Choose Scaffold Analysis Mode",
                [
                    "Selected Molecule Scaffold Explorer",
                    "Full Dataset Scaffold Summary"
                ],
                horizontal=True,
                key="scaffold_analysis_mode"
            )

            # ==================================================
            # MODE 1 — SELECTED MOLECULE SCAFFOLD EXPLORER
            # ==================================================

            if scaffold_mode == "Selected Molecule Scaffold Explorer":

                st.info(
                    "Search or browse the molecule catalog, then select a molecule "
                    "to view its Murcko scaffold and scaffold family."
                )

                full_catalog_csv = molecule_df[
                    [
                        "Molecule_Name",
                        "SMILES"
                    ]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Full Molecule Catalog CSV",
                    data=full_catalog_csv,
                    file_name="scaffold_full_molecule_catalog.csv",
                    mime="text/csv",
                    key="download_scaffold_full_catalog"
                )

                if "scaffold_search_reset_counter" not in st.session_state:
                    st.session_state["scaffold_search_reset_counter"] = 0

                scaffold_search_key = (
                    "scaffold_search_input_"
                    + str(st.session_state["scaffold_search_reset_counter"])
                )

                scaffold_selectbox_key = (
                    "scaffold_selectbox_"
                    + str(st.session_state["scaffold_search_reset_counter"])
                )

                with st.expander(
                    "Search Available Molecule Catalog",
                    expanded=True
                ):

                    col_scaffold_search, col_scaffold_reset = st.columns([5, 1])

                    with col_scaffold_search:

                        scaffold_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=scaffold_search_key,
                            placeholder="Example: benz, ethanol, acid, CCO"
                        )

                    with col_scaffold_reset:

                        st.write("")
                        st.write("")

                        if st.button(
                            "Clear / Reset",
                            key="clear_scaffold_search"
                        ):

                            for key_to_clear in [
                                "scaffold_selected_molecule_name",
                                "scaffold_selected_smiles"
                            ]:
                                if key_to_clear in st.session_state:
                                    del st.session_state[key_to_clear]

                            st.session_state["scaffold_search_reset_counter"] += 1
                            st.rerun()

                    if scaffold_search_query.strip() != "":

                        scaffold_filtered_df = molecule_df[
                            molecule_df["Molecule_Name"].str.contains(
                                scaffold_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                            |
                            molecule_df["SMILES"].str.contains(
                                scaffold_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                        ].copy()

                    else:

                        scaffold_filtered_df = molecule_df.copy()

                    st.info(
                        f"Matching molecules found: {len(scaffold_filtered_df)} "
                        f"out of {len(molecule_df)}"
                    )

                    display_paginated_molecule_table(
                        df=scaffold_filtered_df,
                        table_key="scaffold_filtered_df_catalog",
                        rows_per_page=100,
                        columns=[
                            "Molecule_Name",
                            "SMILES"
                        ]
                    )

                    filtered_scaffold_catalog_csv = scaffold_filtered_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Filtered Molecule List CSV",
                        data=filtered_scaffold_catalog_csv,
                        file_name="scaffold_filtered_molecule_catalog.csv",
                        mime="text/csv",
                        key="download_scaffold_filtered_catalog"
                    )

                    st.markdown("---")

                    if scaffold_filtered_df.empty:

                        st.warning(
                            "No molecule found for this search. Please try another molecule name or SMILES."
                        )

                    else:

                        st.subheader("Select Molecule for Scaffold Analysis")

                        scaffold_selected_index = st.selectbox(
                            "Choose molecule from filtered list",
                            options=scaffold_filtered_df.index,
                            format_func=lambda x: scaffold_filtered_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=scaffold_selectbox_key
                        )

                        scaffold_selected_name = scaffold_filtered_df.loc[
                            scaffold_selected_index,
                            "Molecule_Name"
                        ]

                        scaffold_selected_smiles = scaffold_filtered_df.loc[
                            scaffold_selected_index,
                            "SMILES"
                        ]

                        st.session_state["scaffold_selected_molecule_name"] = (
                            scaffold_selected_name
                        )

                        st.session_state["scaffold_selected_smiles"] = (
                            scaffold_selected_smiles
                        )

                if scaffold_filtered_df.empty:

                    st.stop()

                selected_scaffold_name = st.session_state.get(
                    "scaffold_selected_molecule_name",
                    scaffold_filtered_df.iloc[0]["Molecule_Name"]
                )

                selected_scaffold_smiles = st.session_state.get(
                    "scaffold_selected_smiles",
                    scaffold_filtered_df.iloc[0]["SMILES"]
                )

                st.subheader("Selected Molecule")

                st.success(f"Molecule Name: {selected_scaffold_name}")
                st.success(f"SMILES: {selected_scaffold_smiles}")

                st.text_area(
                    "Copy Selected Molecule Name",
                    value=selected_scaffold_name,
                    height=70,
                    key=f"scaffold_copy_name_{make_safe_filename(selected_scaffold_name)}"
                )

                st.text_area(
                    "Copy Selected SMILES",
                    value=selected_scaffold_smiles,
                    height=70,
                    key=f"scaffold_copy_smiles_{make_safe_filename(selected_scaffold_smiles)}"
                )

                selected_mol = Chem.MolFromSmiles(selected_scaffold_smiles)

                if selected_mol is None:

                    st.error("Invalid SMILES for selected molecule.")

                else:

                    selected_scaffold_smiles_core = get_murcko_scaffold(
                        selected_scaffold_smiles
                    )

                    st.subheader("Molecule Structure vs Murcko Scaffold")

                    col_molecule, col_scaffold = st.columns(2)

                    with col_molecule:

                        st.markdown("### Selected Molecule 2D Structure")

                        selected_molecule_image = Draw.MolToImage(
                            selected_mol,
                            size=(350, 350)
                        )

                        st.image(
                            selected_molecule_image,
                            caption="Selected Molecule"
                        )

                    with col_scaffold:

                        st.markdown("### Murcko Scaffold 2D Structure")

                        if selected_scaffold_smiles_core in [None, "No Scaffold"]:

                            st.warning(
                                "No Murcko scaffold found for this molecule. "
                                "This often happens for acyclic molecules."
                            )

                        else:

                            scaffold_mol = Chem.MolFromSmiles(
                                selected_scaffold_smiles_core
                            )

                            if scaffold_mol is None:

                                st.warning("Scaffold structure could not be rendered.")

                            else:

                                scaffold_image = Draw.MolToImage(
                                    scaffold_mol,
                                    size=(350, 350)
                                )

                                st.image(
                                    scaffold_image,
                                    caption="Murcko Scaffold Core"
                                )

                    st.subheader("Murcko Scaffold SMILES")

                    st.code(
                        f"{selected_scaffold_smiles_core}",
                        language="text"
                    )

                    st.markdown("---")

                    st.subheader("Molecules Sharing the Same Scaffold")

                    scaffold_df = generate_scaffold_dataframe(
                        molecule_df
                    )

                    same_scaffold_df = scaffold_df[
                        scaffold_df["Murcko_Scaffold"]
                        ==
                        selected_scaffold_smiles_core
                    ].copy()

                    if same_scaffold_df.empty:

                        st.info(
                            "No other molecule with the same scaffold was found in the dataset."
                        )

                    else:

                        st.success(
                            f"{len(same_scaffold_df)} molecule(s) share this scaffold."
                        )

                        same_scaffold_display_df = same_scaffold_df[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "Murcko_Scaffold"
                            ]
                        ].copy()

                        st.dataframe(
                            same_scaffold_display_df,
                            width="stretch"
                        )

                        same_scaffold_csv = same_scaffold_display_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Same-Scaffold Molecules CSV",
                            data=same_scaffold_csv,
                            file_name="same_scaffold_molecule_family.csv",
                            mime="text/csv"
                        )

                    st.markdown("---")

                    st.subheader("Scaffold Family Summary")

                    scaffold_summary_df = pd.DataFrame([{
                        "Selected_Molecule_Name": selected_scaffold_name,
                        "Selected_SMILES": selected_scaffold_smiles,
                        "Murcko_Scaffold": selected_scaffold_smiles_core,
                        "Same_Scaffold_Molecule_Count": len(same_scaffold_df)
                    }])

                    st.dataframe(
                        scaffold_summary_df,
                        width="stretch"
                    )

                    scaffold_summary_csv = scaffold_summary_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Selected Molecule Scaffold Summary CSV",
                        data=scaffold_summary_csv,
                        file_name="selected_molecule_scaffold_summary.csv",
                        mime="text/csv"
                    )

            # ==================================================
            # MODE 2 — FULL DATASET SCAFFOLD SUMMARY
            # ==================================================

            else:

                st.info(
                    "Generate full dataset Murcko scaffold statistics and scaffold frequency analysis."
                )

                if st.button(
                    "Generate Full Dataset Scaffold Analysis",
                    key="generate_full_scaffold_analysis"
                ):

                    with st.spinner("Generating Murcko scaffolds for full dataset..."):

                        scaffold_df = generate_scaffold_dataframe(
                            molecule_df
                        )

                    if scaffold_df.empty:

                        st.warning("No valid scaffolds could be generated.")

                    else:

                        st.success("Full dataset scaffold analysis completed.")

                        total_molecules = len(scaffold_df)

                        unique_scaffolds = scaffold_df[
                            "Murcko_Scaffold"
                        ].nunique()

                        no_scaffold_count = len(
                            scaffold_df[
                                scaffold_df["Murcko_Scaffold"] == "No Scaffold"
                            ]
                        )

                        col_s1, col_s2, col_s3 = st.columns(3)

                        with col_s1:
                            st.metric(
                                "Molecules Analyzed",
                                total_molecules
                            )

                        with col_s2:
                            st.metric(
                                "Unique Scaffolds",
                                unique_scaffolds
                            )

                        with col_s3:
                            st.metric(
                                "No Scaffold Molecules",
                                no_scaffold_count
                            )

                        st.subheader("Scaffold Frequency Table")

                        scaffold_freq_df = (
                            scaffold_df
                            .groupby("Murcko_Scaffold")
                            .agg(
                                Molecule_Count=("SMILES", "count"),
                                Example_Molecule=("Molecule_Name", "first"),
                                Example_SMILES=("SMILES", "first")
                            )
                            .reset_index()
                            .sort_values(
                                by="Molecule_Count",
                                ascending=False
                            )
                        )

                        st.dataframe(
                            scaffold_freq_df,
                            width="stretch"
                        )

                        st.subheader("Top Scaffold Frequency Plot")

                        top_n_scaffolds = st.slider(
                            "Select number of top scaffolds to display",
                            min_value=5,
                            max_value=30,
                            value=10,
                            step=5,
                            key="full_scaffold_top_n_slider"
                        )

                        plot_df = scaffold_freq_df.head(
                            top_n_scaffolds
                        ).copy()

                        plot_df["Scaffold_Label"] = (
                            plot_df["Murcko_Scaffold"]
                            .astype(str)
                            .str.slice(0, 25)
                        )

                        fig_scaffold, ax_scaffold = plt.subplots(
                            figsize=(10, 5)
                        )

                        ax_scaffold.bar(
                            plot_df["Scaffold_Label"],
                            plot_df["Molecule_Count"]
                        )

                        ax_scaffold.set_xlabel("Murcko Scaffold")
                        ax_scaffold.set_ylabel("Molecule Count")
                        ax_scaffold.set_title("Top Murcko Scaffold Frequencies")
                        ax_scaffold.tick_params(axis="x", rotation=45)

                        st.pyplot(fig_scaffold)

                        st.subheader("Core Structure Explorer")

                        valid_scaffolds = scaffold_freq_df[
                            scaffold_freq_df["Murcko_Scaffold"] != "No Scaffold"
                        ].copy()

                        if valid_scaffolds.empty:

                            st.info("No core scaffold structures available to visualize.")

                        else:

                            selected_scaffold = st.selectbox(
                                "Select Scaffold Core",
                                options=valid_scaffolds["Murcko_Scaffold"].tolist(),
                                format_func=lambda x: (
                                    f"{x} | Count: "
                                    f"{int(valid_scaffolds.loc[valid_scaffolds['Murcko_Scaffold'] == x, 'Molecule_Count'].iloc[0])}"
                                ),
                                key="full_scaffold_core_selectbox"
                            )

                            scaffold_mol = Chem.MolFromSmiles(
                                selected_scaffold
                            )

                            if scaffold_mol is not None:

                                scaffold_image = Draw.MolToImage(
                                    scaffold_mol,
                                    size=(400, 400)
                                )

                                st.image(
                                    scaffold_image,
                                    caption="Selected Murcko Scaffold Core"
                                )

                                scaffold_img_buffer = BytesIO()
                                scaffold_image.save(
                                    scaffold_img_buffer,
                                    format="PNG"
                                )

                                scaffold_safe_name = make_safe_filename(
                                    selected_scaffold
                                )

                                st.download_button(
                                    label="Download Scaffold Core PNG",
                                    data=scaffold_img_buffer.getvalue(),
                                    file_name=f"{scaffold_safe_name}_scaffold.png",
                                    mime="image/png"
                                )

                                st.subheader("Molecules Belonging to Selected Scaffold")

                                selected_scaffold_molecules = scaffold_df[
                                    scaffold_df["Murcko_Scaffold"] == selected_scaffold
                                ][
                                    [
                                        "Molecule_Name",
                                        "SMILES",
                                        "Murcko_Scaffold"
                                    ]
                                ]

                                st.dataframe(
                                    selected_scaffold_molecules,
                                    width="stretch"
                                )

                        scaffold_csv = scaffold_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Full Scaffold Analysis CSV",
                            data=scaffold_csv,
                            file_name="murcko_scaffold_analysis.csv",
                            mime="text/csv"
                        )

                        scaffold_freq_csv = scaffold_freq_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Scaffold Frequency CSV",
                            data=scaffold_freq_csv,
                            file_name="murcko_scaffold_frequency.csv",
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(f"Scaffold analysis failed: {e}")


    with tab8:

        st.subheader("Out-of-Distribution (OOD) Detection")

        st.write(
            "Detect whether a molecule is chemically similar to the known dataset chemistry. "
            "This helps identify when a melting point prediction may be unreliable."
        )

        try:

            ood_df = load_molecule_dataset()

            ood_input_mode = st.radio(
                "Choose OOD Input Method",
                [
                    "Select from Dataset",
                    "Enter Custom SMILES"
                ],
                horizontal=True,
                key="ood_input_mode"
            )

            ood_query_name = "Custom Input"
            ood_query_smiles = "CCO"

            # ==================================================
            # SELECT FROM DATASET WITH SEARCHABLE CATALOG
            # ==================================================

            if ood_input_mode == "Select from Dataset":

                st.info(
                    "Search or browse the available molecule catalog, then select a molecule "
                    "for OOD detection."
                )

                ood_catalog_csv = ood_df[
                    [
                        "Molecule_Name",
                        "SMILES"
                    ]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Full Molecule Catalog CSV",
                    data=ood_catalog_csv,
                    file_name="ood_full_molecule_catalog.csv",
                    mime="text/csv",
                    key="download_ood_full_catalog"
                )

                if "ood_search_reset_counter" not in st.session_state:
                    st.session_state["ood_search_reset_counter"] = 0

                ood_search_key = (
                    "ood_search_input_"
                    + str(st.session_state["ood_search_reset_counter"])
                )

                ood_selectbox_key = (
                    "ood_selectbox_"
                    + str(st.session_state["ood_search_reset_counter"])
                )

                with st.expander(
                    "Search Available Molecule Catalog for OOD Detection",
                    expanded=True
                ):

                    col_ood_search, col_ood_reset = st.columns([5, 1])

                    with col_ood_search:

                        ood_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=ood_search_key,
                            placeholder="Example: ethanol, benz, acid, CCO"
                        )

                    with col_ood_reset:

                        st.write("")
                        st.write("")

                        if st.button(
                            "Clear / Reset",
                            key="clear_ood_search"
                        ):

                            for key_to_clear in [
                                "ood_selected_molecule_name",
                                "ood_selected_smiles"
                            ]:
                                if key_to_clear in st.session_state:
                                    del st.session_state[key_to_clear]

                            st.session_state["ood_search_reset_counter"] += 1
                            st.rerun()

                    if ood_search_query.strip() != "":

                        ood_filtered_df = ood_df[
                            ood_df["Molecule_Name"].str.contains(
                                ood_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                            |
                            ood_df["SMILES"].str.contains(
                                ood_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                        ].copy()

                    else:

                        ood_filtered_df = ood_df.copy()

                    st.info(
                        f"Matching molecules found: {len(ood_filtered_df)} "
                        f"out of {len(ood_df)}"
                    )

                    display_paginated_molecule_table(
                        df=ood_filtered_df,
                        table_key="ood_filtered_df_catalog",
                        rows_per_page=100,
                        columns=[
                            "Molecule_Name",
                            "SMILES"
                        ]
                    )

                    filtered_ood_csv = ood_filtered_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Filtered OOD Molecule List CSV",
                        data=filtered_ood_csv,
                        file_name="ood_filtered_molecule_catalog.csv",
                        mime="text/csv",
                        key="download_ood_filtered_catalog"
                    )

                    st.markdown("---")

                    if ood_filtered_df.empty:

                        st.warning(
                            "No molecule found for this search. Please try another molecule name or SMILES."
                        )

                    else:

                        st.subheader("Select Molecule for OOD Detection")

                        selected_ood_index = st.selectbox(
                            "Choose molecule from filtered list",
                            options=ood_filtered_df.index,
                            format_func=lambda x: ood_filtered_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=ood_selectbox_key
                        )

                        ood_query_name = ood_filtered_df.loc[
                            selected_ood_index,
                            "Molecule_Name"
                        ]

                        ood_query_smiles = ood_filtered_df.loc[
                            selected_ood_index,
                            "SMILES"
                        ]

                        st.session_state["ood_selected_molecule_name"] = (
                            ood_query_name
                        )

                        st.session_state["ood_selected_smiles"] = (
                            ood_query_smiles
                        )

                        st.success(
                            "Selected molecule is now added to the OOD detection input."
                        )

                if "ood_filtered_df" in locals() and not ood_filtered_df.empty:

                    ood_query_name = st.session_state.get(
                        "ood_selected_molecule_name",
                        ood_filtered_df.iloc[0]["Molecule_Name"]
                    )

                    ood_query_smiles = st.session_state.get(
                        "ood_selected_smiles",
                        ood_filtered_df.iloc[0]["SMILES"]
                    )

                else:

                    ood_query_name = "Dataset Selection"
                    ood_query_smiles = "CCO"

            # ==================================================
            # CUSTOM SMILES INPUT
            # ==================================================

            else:

                ood_query_name = "Custom Input"

                ood_query_smiles = st.text_input(
                    "Enter SMILES for OOD Detection",
                    value="CCO",
                    key="ood_custom_smiles"
                )

            st.subheader("Current Molecule Selected for OOD Detection")

            st.success(f"Molecule: {ood_query_name}")
            st.success(f"SMILES: {ood_query_smiles}")

            st.text_area(
                "Copy OOD Molecule Name",
                value=ood_query_name,
                height=70,
                key=f"ood_copy_name_{make_safe_filename(ood_query_name)}"
            )

            st.text_area(
                "Copy OOD SMILES",
                value=ood_query_smiles,
                height=70,
                key=f"ood_copy_smiles_{make_safe_filename(ood_query_smiles)}"
            )

            # ==================================================
            # RUN OOD DETECTION
            # ==================================================

            if st.button(
                "Run OOD Detection",
                key="run_ood_detection"
            ):

                with st.spinner(
                    "Comparing molecule against known dataset chemistry..."
                ):

                    ood_result = detect_ood_molecule(
                        ood_query_smiles,
                        ood_df
                    )

                st.subheader("OOD Detection Result")

                ood_result_df = pd.DataFrame([ood_result])
                st.dataframe(
                    ood_result_df,
                    width="stretch"
                )

                max_similarity = ood_result[
                    "Max_Tanimoto_Similarity"
                ]

                col1, col2, col3 = st.columns(3)

                with col1:
                    if max_similarity is not None:
                        st.metric(
                            "Nearest Similarity",
                            f"{max_similarity:.4f}"
                        )
                    else:
                        st.metric(
                            "Nearest Similarity",
                            "N/A"
                        )

                with col2:
                    st.metric(
                        "OOD Status",
                        ood_result["OOD_Status"]
                    )

                with col3:
                    st.metric(
                        "Reliability",
                        ood_result["Reliability"]
                    )

                if ood_result["OOD_Status"] == "In Distribution":

                    st.success(
                        ood_result["Warning"]
                    )

                elif ood_result["OOD_Status"] == "Borderline":

                    st.warning(
                        ood_result["Warning"]
                    )

                else:

                    st.error(
                        ood_result["Warning"]
                    )

                st.markdown("---")

                st.subheader("Top 5 Nearest Molecules")

                similarity_df = calculate_all_similarity_scores(
                    ood_query_smiles,
                    ood_df
                )

                if similarity_df.empty:

                    st.warning(
                        "Similarity scores could not be calculated."
                    )

                else:

                    top_5_nearest_df = similarity_df.head(5).copy()

                    st.dataframe(
                        top_5_nearest_df,
                        width="stretch"
                    )

                    top_5_csv = top_5_nearest_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Top 5 Nearest Molecules CSV",
                        data=top_5_csv,
                        file_name="ood_top_5_nearest_molecules.csv",
                        mime="text/csv"
                    )

                    st.subheader("Similarity Histogram")

                    fig_similarity, ax_similarity = plt.subplots(
                        figsize=(8, 4)
                    )

                    ax_similarity.hist(
                        similarity_df["Tanimoto_Similarity"],
                        bins=30
                    )

                    ax_similarity.axvline(
                        x=0.70,
                        linestyle="--",
                        label="In Distribution Threshold"
                    )

                    ax_similarity.axvline(
                        x=0.40,
                        linestyle="--",
                        label="OOD Threshold"
                    )

                    ax_similarity.set_xlabel("Tanimoto Similarity")
                    ax_similarity.set_ylabel("Number of Dataset Molecules")
                    ax_similarity.set_title(
                        "Similarity Distribution Against Full Dataset"
                    )
                    ax_similarity.legend()

                    st.pyplot(fig_similarity)

                    st.info(
                        "Interpretation: if the query molecule has very few high-similarity "
                        "neighbors, it may be less represented in the training chemistry. "
                        "Dense similarity near the right side indicates stronger dataset support."
                    )

                    full_similarity_csv = similarity_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Full Similarity Scores CSV",
                        data=full_similarity_csv,
                        file_name="ood_full_similarity_scores.csv",
                        mime="text/csv"
                    )

                    st.markdown("---")

                    st.subheader("Applicability Domain Boundary")

                    applicability_threshold = st.slider(
                        "Tanimoto Applicability Domain Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.40,
                        step=0.01,
                        key="ood_applicability_threshold"
                    )

                    nearest_similarity_value = top_5_nearest_df[
                        "Tanimoto_Similarity"
                    ].iloc[0]

                    col_ad1, col_ad2, col_ad3 = st.columns(3)

                    with col_ad1:
                        st.metric(
                            "Nearest Similarity",
                            f"{nearest_similarity_value:.4f}"
                        )

                    with col_ad2:
                        st.metric(
                            "AD Threshold",
                            f"{applicability_threshold:.2f}"
                        )

                    with col_ad3:
                        if nearest_similarity_value >= applicability_threshold:
                            st.metric(
                                "Applicability Domain",
                                "Inside"
                            )
                        else:
                            st.metric(
                                "Applicability Domain",
                                "Outside"
                            )

                    if nearest_similarity_value >= applicability_threshold:
                        st.success(
                            "The molecule is inside the similarity-based applicability domain."
                        )
                    else:
                        st.error(
                            "The molecule is outside the similarity-based applicability domain. "
                            "Prediction reliability may be lower."
                        )

                    st.info(
                        "Applicability domain boundary is estimated using the highest "
                        "Tanimoto similarity between the query molecule and the known dataset."
                    )

                    st.markdown("---")

                    st.subheader("Mahalanobis Latent Distance")

                    st.write(
                        "This computes a PCA-based latent chemical-space distance using Morgan fingerprints. "
                        "Higher distance means the molecule is farther from the dataset chemical-space center."
                    )

                    mahal_distance, mahal_95, mahal_99 = (
                        calculate_mahalanobis_distance_for_smiles(
                            ood_query_smiles,
                            ood_df,
                            max_reference_molecules=1000,
                            n_bits=2048
                        )
                    )

                    if mahal_distance is None:

                        st.warning(
                            "Mahalanobis distance could not be calculated for this molecule."
                        )

                    else:

                        col_m1, col_m2, col_m3 = st.columns(3)

                        with col_m1:
                            st.metric(
                                "Mahalanobis Distance",
                                f"{mahal_distance:.4f}"
                            )

                        with col_m2:
                            st.metric(
                                "95% Dataset Boundary",
                                f"{mahal_95:.4f}"
                            )

                        with col_m3:
                            st.metric(
                                "99% Dataset Boundary",
                                f"{mahal_99:.4f}"
                            )

                        if mahal_distance <= mahal_95:

                            st.success(
                                "Mahalanobis result: molecule is inside the 95% latent chemical-space boundary."
                            )

                        elif mahal_distance <= mahal_99:

                            st.warning(
                                "Mahalanobis result: molecule is between the 95% and 99% boundary. "
                                "Use prediction with caution."
                            )

                        else:

                            st.error(
                                "Mahalanobis result: molecule is outside the 99% latent boundary. "
                                "This indicates possible out-of-distribution chemistry."
                            )

                        mahalanobis_summary_df = pd.DataFrame([{
                            "Query_SMILES": ood_query_smiles,
                            "Mahalanobis_Distance": mahal_distance,
                            "Boundary_95": mahal_95,
                            "Boundary_99": mahal_99,
                            "Latent_Domain_Status": (
                                "Inside 95% Boundary"
                                if mahal_distance <= mahal_95
                                else (
                                    "Between 95% and 99% Boundary"
                                    if mahal_distance <= mahal_99
                                    else "Outside 99% Boundary"
                                )
                            )
                        }])

                        st.dataframe(
                            mahalanobis_summary_df,
                            width="stretch"
                        )

                        mahalanobis_csv = mahalanobis_summary_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Mahalanobis OOD Summary CSV",
                            data=mahalanobis_csv,
                            file_name="ood_mahalanobis_summary.csv",
                            mime="text/csv"
                        )

                st.markdown("---")

                st.subheader("Advanced Prediction Reliability inside OOD")

                st.write(
                    "This section combines prediction uncertainty and prediction interval analysis "
                    "with OOD detection. It does not use any 3D molecular visualization."
                )

                try:

                    rdkit_pred_ood = float(
                        predict_melting_point(ood_query_smiles)
                    )

                    hybrid_pred_ood = float(
                        predict_hybrid_gat(ood_query_smiles)
                    )

                    ensemble_pred_ood = (
                        0.4 * rdkit_pred_ood
                        +
                        0.6 * hybrid_pred_ood
                    )

                    uncertainty_ood = calculate_prediction_uncertainty(
                        rdkit_pred_ood,
                        hybrid_pred_ood
                    )

                    deep_ensemble_ood = calculate_deep_ensemble_uncertainty(
                        rdkit_pred_ood,
                        hybrid_pred_ood,
                        ensemble_pred_ood
                    )

                    conformal_ood = calculate_conformal_prediction_interval(
                        prediction_k=ensemble_pred_ood,
                        uncertainty_range=uncertainty_ood[
                            "uncertainty_range"
                        ],
                        confidence_label=uncertainty_ood[
                            "confidence_label"
                        ]
                    )

                    st.subheader("Deep Ensemble Uncertainty inside OOD")

                    col_de1, col_de2, col_de3, col_de4 = st.columns(4)

                    with col_de1:
                        st.metric(
                            "RDKit Prediction",
                            f"{rdkit_pred_ood:.2f} K"
                        )

                    with col_de2:
                        st.metric(
                            "Hybrid GAT Prediction",
                            f"{hybrid_pred_ood:.2f} K"
                        )

                    with col_de3:
                        st.metric(
                            "Ensemble Prediction",
                            f"{ensemble_pred_ood:.2f} K"
                        )

                    with col_de4:
                        st.metric(
                            "Prediction STD",
                            f"± {deep_ensemble_ood['STD_Prediction_K']:.2f} K"
                        )

                    deep_ensemble_ood_df = pd.DataFrame([
                        deep_ensemble_ood
                    ])

                    st.dataframe(
                        deep_ensemble_ood_df,
                        width="stretch"
                    )

                    if deep_ensemble_ood["Uncertainty_Label"] == "Low Uncertainty":

                        st.success(
                            f"Deep ensemble result: "
                            f"{deep_ensemble_ood['Uncertainty_Label']}"
                        )

                    elif deep_ensemble_ood["Uncertainty_Label"] == "Moderate Uncertainty":

                        st.warning(
                            f"Deep ensemble result: "
                            f"{deep_ensemble_ood['Uncertainty_Label']}"
                        )

                    else:

                        st.error(
                            f"Deep ensemble result: "
                            f"{deep_ensemble_ood['Uncertainty_Label']}"
                        )

                    st.info(
                        "Deep ensemble uncertainty is estimated from disagreement among "
                        "the RDKit model, Hybrid GAT model, and final ensemble prediction."
                    )

                    st.markdown("---")

                    st.subheader("Conformal Prediction Interval inside OOD")

                    col_cf1, col_cf2, col_cf3 = st.columns(3)

                    with col_cf1:
                        st.metric(
                            "Lower Bound",
                            f"{conformal_ood['Lower_Bound_K']:.2f} K"
                        )

                    with col_cf2:
                        st.metric(
                            "Upper Bound",
                            f"{conformal_ood['Upper_Bound_K']:.2f} K"
                        )

                    with col_cf3:
                        st.metric(
                            "Interval Type",
                            conformal_ood["Interval_Label"]
                        )

                    st.success(
                        f"Approximate prediction interval: "
                        f"{conformal_ood['Lower_Bound_K']:.2f} K to "
                        f"{conformal_ood['Upper_Bound_K']:.2f} K "
                        f"({conformal_ood['Lower_Bound_C']:.2f} °C to "
                        f"{conformal_ood['Upper_Bound_C']:.2f} °C)"
                    )

                    conformal_ood_df = pd.DataFrame([
                        conformal_ood
                    ])

                    st.dataframe(
                        conformal_ood_df,
                        width="stretch"
                    )

                    reliability_summary_df = pd.DataFrame([{
                        "Query_SMILES": ood_query_smiles,
                        "RDKit_Prediction_K": round(rdkit_pred_ood, 2),
                        "Hybrid_GAT_Prediction_K": round(hybrid_pred_ood, 2),
                        "Ensemble_Prediction_K": round(ensemble_pred_ood, 2),
                        "Ensemble_Prediction_C": round(ensemble_pred_ood - 273.15, 2),
                        "Deep_Ensemble_STD_K": deep_ensemble_ood[
                            "STD_Prediction_K"
                        ],
                        "Deep_Ensemble_Label": deep_ensemble_ood[
                            "Uncertainty_Label"
                        ],
                        "Confidence_%": uncertainty_ood[
                            "confidence"
                        ],
                        "Confidence_Label": uncertainty_ood[
                            "confidence_label"
                        ],
                        "Conformal_Lower_K": conformal_ood[
                            "Lower_Bound_K"
                        ],
                        "Conformal_Upper_K": conformal_ood[
                            "Upper_Bound_K"
                        ],
                        "Conformal_Interval_Label": conformal_ood[
                            "Interval_Label"
                        ]
                    }])

                    reliability_csv = reliability_summary_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download OOD Prediction Reliability CSV",
                        data=reliability_csv,
                        file_name="ood_prediction_reliability_summary.csv",
                        mime="text/csv"
                    )

                except Exception as e:

                    st.warning(
                        f"Advanced prediction reliability could not be calculated: {e}"
                    )

                st.markdown("---")

                st.subheader("Applicability Boundary Visualization")

                try:

                    boundary_threshold = st.slider(
                        "Applicability Boundary Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.40,
                        step=0.01,
                        key="ood_boundary_gauge_threshold"
                    )

                    nearest_similarity_for_gauge = float(
                        top_5_nearest_df[
                            "Tanimoto_Similarity"
                        ].iloc[0]
                    )

                    gauge_fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=nearest_similarity_for_gauge,
                            title={
                                "text": "Nearest Similarity vs Applicability Boundary"
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 1]
                                },
                                "threshold": {
                                    "line": {
                                        "color": "red",
                                        "width": 4
                                    },
                                    "thickness": 0.75,
                                    "value": boundary_threshold
                                },
                                "steps": [
                                    {
                                        "range": [0, boundary_threshold],
                                        "color": "lightgray"
                                    },
                                    {
                                        "range": [boundary_threshold, 1],
                                        "color": "lightgreen"
                                    }
                                ]
                            }
                        )
                    )

                    gauge_fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        gauge_fig,
                        width="stretch"
                    )

                    if nearest_similarity_for_gauge >= boundary_threshold:

                        st.success(
                            "Applicability visualization: molecule is inside the "
                            "similarity-based applicability domain."
                        )

                    else:

                        st.error(
                            "Applicability visualization: molecule is outside the "
                            "similarity-based applicability domain."
                        )

                except Exception as e:

                    st.warning(
                        f"Applicability boundary visualization failed: {e}"
                    )

                st.markdown("---")

                st.subheader("Interactive Similarity Network Graph")

                try:

                    network_df = top_5_nearest_df.copy()

                    graph = nx.Graph()

                    query_node = "Query Molecule"

                    graph.add_node(
                        query_node,
                        node_type="Query"
                    )

                    for _, network_row in network_df.iterrows():

                        molecule_name = str(
                            network_row["Molecule_Name"]
                        )

                        similarity_value = float(
                            network_row["Tanimoto_Similarity"]
                        )

                        label = (
                            molecule_name[:40]
                            if molecule_name != "Name Not Found"
                            else network_row["SMILES"][:40]
                        )

                        graph.add_node(
                            label,
                            node_type="Dataset",
                            similarity=similarity_value
                        )

                        graph.add_edge(
                            query_node,
                            label,
                            weight=similarity_value
                        )

                    positions = nx.spring_layout(
                        graph,
                        seed=42,
                        k=0.8
                    )

                    edge_x = []
                    edge_y = []

                    for edge in graph.edges():

                        x0, y0 = positions[edge[0]]
                        x1, y1 = positions[edge[1]]

                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        line=dict(
                            width=1.5
                        ),
                        hoverinfo="none",
                        mode="lines"
                    )

                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []

                    for node in graph.nodes():

                        x, y = positions[node]

                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)

                        if node == query_node:
                            node_size.append(28)
                        else:
                            sim = graph.nodes[node].get(
                                "similarity",
                                0.5
                            )
                            node_size.append(
                                15 + sim * 20
                            )

                    node_trace = go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_text,
                        textposition="top center",
                        hoverinfo="text",
                        marker=dict(
                            size=node_size,
                            line=dict(
                                width=1
                            )
                        )
                    )

                    network_fig = go.Figure(
                        data=[
                            edge_trace,
                            node_trace
                        ]
                    )

                    network_fig.update_layout(
                        title="Query Molecule Similarity Network: Top 5 Nearest Molecules",
                        showlegend=False,
                        height=550,
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20
                        ),
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False
                        )
                    )

                    st.plotly_chart(
                        network_fig,
                        width="stretch"
                    )

                    st.info(
                        "The center node is the query molecule. Connected nodes are "
                        "the top 5 most similar dataset molecules. Larger dataset nodes "
                        "represent higher Tanimoto similarity."
                    )

                except Exception as e:

                    st.warning(
                        f"Similarity network graph failed: {e}"
                    )


                st.markdown("---")

                st.subheader("PCA Chemical-Space Visualization inside OOD")

                st.write(
                    "This PCA plot shows where the query molecule lies compared with "
                    "reference dataset molecules in Morgan fingerprint chemical space."
                )

                try:

                    pca_ood_df = generate_ood_chemical_space_embeddings(
                        query_smiles=ood_query_smiles,
                        molecule_df=ood_df,
                        max_reference_molecules=1000,
                        n_bits=2048,
                        method="PCA"
                    )

                    if pca_ood_df.empty:

                        st.warning(
                            "PCA OOD visualization could not be generated."
                        )

                    else:

                        pca_fig = px.scatter(
                            pca_ood_df,
                            x="X",
                            y="Y",
                            color="Point_Type",
                            hover_data=[
                                "Molecule_Name",
                                "SMILES"
                            ],
                            title="OOD PCA Chemical-Space Visualization"
                        )

                        pca_fig.update_traces(
                            marker=dict(
                                size=7,
                                opacity=0.75
                            )
                        )

                        st.plotly_chart(
                            pca_fig,
                            width="stretch"
                        )

                        pca1_var = pca_ood_df[
                            "Explained_Variance_PC1"
                        ].iloc[0]

                        pca2_var = pca_ood_df[
                            "Explained_Variance_PC2"
                        ].iloc[0]

                        st.success(
                            f"PCA explained variance: "
                            f"PC1 = {pca1_var}% | PC2 = {pca2_var}%"
                        )

                        pca_ood_csv = pca_ood_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download OOD PCA Coordinates CSV",
                            data=pca_ood_csv,
                            file_name="ood_pca_chemical_space.csv",
                            mime="text/csv"
                        )

                except Exception as e:

                    st.warning(
                        f"PCA OOD visualization failed: {e}"
                    )

                st.markdown("---")

                st.subheader("UMAP Nearest-Neighbor Visualization inside OOD")

                st.write(
                    "This UMAP plot shows local neighborhood structure around the "
                    "query molecule in chemical space. The query point is highlighted "
                    "against reference dataset molecules."
                )

                if not UMAP_AVAILABLE:

                    st.warning(
                        "UMAP is not installed. Please install it first:"
                    )

                    st.code(
                        "pip install umap-learn",
                        language="bash"
                    )

                else:

                    try:

                        umap_ood_df = generate_ood_chemical_space_embeddings(
                            query_smiles=ood_query_smiles,
                            molecule_df=ood_df,
                            max_reference_molecules=1000,
                            n_bits=2048,
                            method="UMAP"
                        )

                        if umap_ood_df.empty:

                            st.warning(
                                "UMAP OOD visualization could not be generated."
                            )

                        else:

                            umap_fig = px.scatter(
                                umap_ood_df,
                                x="X",
                                y="Y",
                                color="Point_Type",
                                hover_data=[
                                    "Molecule_Name",
                                    "SMILES"
                                ],
                                title="OOD UMAP Nearest-Neighbor Chemical-Space Visualization"
                            )

                            umap_fig.update_traces(
                                marker=dict(
                                    size=7,
                                    opacity=0.75
                                )
                            )

                            st.plotly_chart(
                                umap_fig,
                                width="stretch"
                            )

                            st.info(
                                "Interpretation: if the query molecule appears isolated "
                                "from dense dataset regions, it may represent less familiar "
                                "or out-of-distribution chemistry."
                            )

                            umap_ood_csv = umap_ood_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download OOD UMAP Coordinates CSV",
                                data=umap_ood_csv,
                                file_name="ood_umap_chemical_space.csv",
                                mime="text/csv"
                            )

                    except Exception as e:

                        st.warning(
                            f"UMAP OOD visualization failed: {e}"
                        )

                st.markdown("---")

                st.subheader("Query Molecule vs Nearest Dataset Molecule")

                query_mol = Chem.MolFromSmiles(
                    ood_query_smiles
                )

                nearest_smiles = ood_result[
                    "Nearest_SMILES"
                ]

                nearest_name = ood_result[
                    "Nearest_Molecule_Name"
                ]

                col_query, col_nearest = st.columns(2)

                with col_query:

                    st.markdown("### Query Molecule")

                    if query_mol is not None:

                        query_image = Draw.MolToImage(
                            query_mol,
                            size=(330, 330)
                        )

                        st.image(
                            query_image,
                            caption=ood_query_name
                        )

                    else:

                        st.warning("Query molecule image could not be generated.")

                with col_nearest:

                    st.markdown("### Nearest Dataset Molecule")

                    if nearest_smiles is not None:

                        nearest_mol = Chem.MolFromSmiles(
                            nearest_smiles
                        )

                        if nearest_mol is not None:

                            nearest_image = Draw.MolToImage(
                                nearest_mol,
                                size=(330, 330)
                            )

                            st.image(
                                nearest_image,
                                caption=f"Nearest Molecule: {nearest_name}"
                            )

                            st.code(
                                f"Nearest Molecule: {nearest_name}\n"
                                f"SMILES: {nearest_smiles}",
                                language="text"
                            )

                    else:

                        st.warning("Nearest molecule could not be identified.")

                st.markdown("---")

                st.subheader("Prediction Reliability Guidance")

                guidance_df = pd.DataFrame({
                    "Similarity Range": [
                        ">= 0.70",
                        "0.40 to 0.70",
                        "< 0.40"
                    ],
                    "OOD Status": [
                        "In Distribution",
                        "Borderline",
                        "Out of Distribution"
                    ],
                    "Prediction Guidance": [
                        "Prediction is more reliable because molecule is chemically similar to known dataset chemistry.",
                        "Use prediction carefully. Molecule is only moderately similar to known dataset chemistry.",
                        "Prediction may be unreliable. Molecule is unlike known dataset chemistry."
                    ]
                })

                st.dataframe(
                    guidance_df,
                    width="stretch"
                )

                ood_csv = ood_result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download OOD Detection CSV",
                    data=ood_csv,
                    file_name="ood_detection_result.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"OOD detection failed: {e}")


    with tab9:

        st.subheader("PCA Chemical Space Visualization")

        st.write(
            "Interactive PCA visualization of molecular chemical space using Morgan fingerprints. "
            "Use this tab for both single molecule PCA search and complete dataset PCA exploration."
        )

        try:

            pca_full_df = load_molecule_dataset()

            pca_mode = st.radio(
                "Choose PCA Mode",
                [
                    "Single Molecule PCA Search",
                    "Complete Dataset PCA Visualization"
                ],
                horizontal=True,
                key="pca_visualization_mode"
            )

            selected_pca_name = None
            selected_pca_smiles = None

            if pca_mode == "Single Molecule PCA Search":

                st.info(
                    "Search/select one molecule. The selected molecule will be highlighted on the PCA plot."
                )

                if "pca_search_reset_counter" not in st.session_state:
                    st.session_state["pca_search_reset_counter"] = 0

                pca_search_key = (
                    "pca_single_search_"
                    + str(st.session_state["pca_search_reset_counter"])
                )

                pca_select_key = (
                    "pca_single_select_"
                    + str(st.session_state["pca_search_reset_counter"])
                )

                col_pca_search, col_pca_reset = st.columns([5, 1])

                with col_pca_search:

                    pca_search_query = st.text_input(
                        "Search molecule by IUPAC/name or SMILES",
                        value="",
                        key=pca_search_key,
                        placeholder="Example: ethanol, benz, acid, CCO"
                    )

                with col_pca_reset:

                    st.write("")
                    st.write("")

                    if st.button(
                        "Clear / Reset",
                        key="clear_pca_single_search"
                    ):

                        for key_to_clear in [
                            "pca_selected_molecule_name",
                            "pca_selected_smiles"
                        ]:
                            if key_to_clear in st.session_state:
                                del st.session_state[key_to_clear]

                        st.session_state["pca_search_reset_counter"] += 1

                        st.rerun()

                if pca_search_query.strip() != "":

                    pca_filtered_df = pca_full_df[
                        pca_full_df["Molecule_Name"].str.contains(
                            pca_search_query,
                            case=False,
                            na=False,
                            regex=False
                        )
                        |
                        pca_full_df["SMILES"].str.contains(
                            pca_search_query,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ].copy()

                else:

                    pca_filtered_df = pca_full_df.copy()

                st.info(
                    f"Matching molecules found: {len(pca_filtered_df)} "
                    f"out of {len(pca_full_df)}"
                )

                display_paginated_molecule_table(


                    df=pca_filtered_df,


                    table_key="pca_filtered_df_catalog",


                    rows_per_page=100,


                    columns=[


                        "Molecule_Name",


                        "SMILES"


                    ]


                )

                if pca_filtered_df.empty:

                    st.warning("No molecule found. Please try another search.")

                else:

                    selected_pca_index = st.selectbox(
                        "Select molecule to highlight on PCA plot",
                        options=pca_filtered_df.index,
                        format_func=lambda x: pca_filtered_df.loc[
                            x,
                            "Molecule_Display"
                        ],
                        key=pca_select_key
                    )

                    selected_pca_name = pca_filtered_df.loc[
                        selected_pca_index,
                        "Molecule_Name"
                    ]

                    selected_pca_smiles = pca_filtered_df.loc[
                        selected_pca_index,
                        "SMILES"
                    ]

                    st.session_state["pca_selected_molecule_name"] = (
                        selected_pca_name
                    )

                    st.session_state["pca_selected_smiles"] = (
                        selected_pca_smiles
                    )

                    st.success(f"Selected Molecule: {selected_pca_name}")
                    st.success(f"Selected SMILES: {selected_pca_smiles}")

            st.markdown("---")

            sample_size_pca = st.slider(
                "Number of molecules to visualize",
                min_value=100,
                max_value=min(3000, len(pca_full_df)),
                value=min(1000, len(pca_full_df)),
                step=100,
                key="interactive_pca_sample_size"
            )

            pca_random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="interactive_pca_random_seed"
            )

            color_option_pca = st.selectbox(
                "Color points by",
                [
                    "Point Type",
                    "Scaffold Group",
                    "Outlier Status",
                    "PCA OOD Status",
                    "Molecule Name"
                ],
                key="interactive_pca_color_option"
            )

            if st.button(
                "Generate Interactive PCA Chemical Space Plot",
                key="generate_interactive_pca_plot"
            ):

                with st.spinner("Generating interactive PCA chemical space..."):

                    pca_df, pca_model = generate_interactive_pca_chemical_space(
                        molecule_df=pca_full_df,
                        selected_smiles=selected_pca_smiles,
                        sample_size=sample_size_pca,
                        random_state=int(pca_random_seed),
                        n_bits=2048
                    )

                if pca_df.empty:

                    st.warning(
                        "PCA visualization could not be generated. Not enough valid molecules."
                    )

                else:

                    st.success("Interactive PCA chemical space generated successfully.")

                    pc1_var = pca_df["Explained_Variance_PC1_%"].iloc[0]
                    pc2_var = pca_df["Explained_Variance_PC2_%"].iloc[0]

                    col_pca1, col_pca2, col_pca3 = st.columns(3)

                    with col_pca1:
                        st.metric(
                            "PC1 Explained Variance",
                            f"{pc1_var:.2f}%"
                        )

                    with col_pca2:
                        st.metric(
                            "PC2 Explained Variance",
                            f"{pc2_var:.2f}%"
                        )

                    with col_pca3:
                        st.metric(
                            "Total Explained Variance",
                            f"{pc1_var + pc2_var:.2f}%"
                        )

                    if color_option_pca == "Point Type":
                        color_column = "Point_Type"
                    elif color_option_pca == "Scaffold Group":
                        color_column = "Scaffold_Group"
                    elif color_option_pca == "Outlier Status":
                        color_column = "Outlier_Status"
                    elif color_option_pca == "PCA OOD Status":
                        color_column = "PCA_OOD_Status"
                    else:
                        color_column = "Molecule_Name"

                    st.subheader("Interactive PCA Chemical Space Plot")

                    pca_fig = px.scatter(
                        pca_df,
                        x="PCA_1",
                        y="PCA_2",
                        color=color_column,
                        hover_data=[
                            "Molecule_Name",
                            "SMILES",
                            "Point_Type",
                            "Murcko_Scaffold",
                            "PCA_Distance_From_Center",
                            "Outlier_Status",
                            "PCA_OOD_Status"
                        ],
                        title="Interactive PCA Chemical Space Visualization"
                    )

                    pca_fig.update_traces(
                        marker=dict(
                            size=7,
                            opacity=0.75
                        )
                    )

                    selected_rows = pca_df[
                        pca_df["Point_Type"] == "Selected Molecule"
                    ]

                    if not selected_rows.empty:

                        pca_fig.add_scatter(
                            x=selected_rows["PCA_1"],
                            y=selected_rows["PCA_2"],
                            mode="markers+text",
                            text=["Selected Molecule"],
                            textposition="top center",
                            marker=dict(
                                size=18,
                                symbol="star",
                                line=dict(
                                    width=2
                                )
                            ),
                            name="Selected Molecule Highlight"
                        )

                    st.plotly_chart(
                        pca_fig,
                        width="stretch"
                    )

                    st.markdown("---")

                    if not selected_rows.empty:

                        st.subheader("Selected Molecule PCA Location")

                        selected_location_df = selected_rows[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "PCA_1",
                                "PCA_2",
                                "PCA_Distance_From_Center",
                                "Outlier_Status",
                                "PCA_OOD_Status",
                                "Murcko_Scaffold"
                            ]
                        ].copy()

                        st.dataframe(
                            selected_location_df,
                            width="stretch"
                        )

                        selected_status = selected_location_df[
                            "PCA_OOD_Status"
                        ].iloc[0]

                        if selected_status == "Inside PCA Space":

                            st.success(
                                "Selected molecule is located inside the main PCA chemical space."
                            )

                        else:

                            st.warning(
                                "Selected molecule is relatively far from the PCA center and may behave like an outlier."
                            )

                    st.subheader("PCA Outlier Detection Table")

                    outlier_df = pca_df[
                        pca_df["Outlier_Status"] == "Potential Outlier"
                    ].sort_values(
                        by="PCA_Distance_From_Center",
                        ascending=False
                    )[
                        [
                            "Molecule_Name",
                            "SMILES",
                            "PCA_1",
                            "PCA_2",
                            "PCA_Distance_From_Center",
                            "Outlier_Status",
                            "PCA_OOD_Status"
                        ]
                    ]

                    if outlier_df.empty:

                        st.info("No PCA outliers detected.")

                    else:

                        st.warning(
                            f"{len(outlier_df)} potential PCA outlier(s) detected."
                        )

                        st.dataframe(
                            outlier_df.head(50),
                            width="stretch"
                        )

                        outlier_csv = outlier_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download PCA Outlier Table CSV",
                            data=outlier_csv,
                            file_name="pca_outlier_detection_table.csv",
                            mime="text/csv"
                        )

                    st.subheader("Download PCA Coordinates")

                    download_cols = [
                        "Molecule_Name",
                        "SMILES",
                        "PCA_1",
                        "PCA_2",
                        "Point_Type",
                        "Murcko_Scaffold",
                        "Scaffold_Group",
                        "PCA_Distance_From_Center",
                        "Outlier_Status",
                        "PCA_OOD_Status",
                        "Explained_Variance_PC1_%",
                        "Explained_Variance_PC2_%"
                    ]

                    pca_csv = pca_df[download_cols].to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download PCA Coordinates CSV",
                        data=pca_csv,
                        file_name="interactive_pca_chemical_space_coordinates.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Interpretation: PCA shows global molecular distribution. "
                        "Points far from dense regions or flagged as potential outliers "
                        "may represent unusual chemistry compared with the dataset."
                    )

        except Exception as e:

            st.error(f"PCA chemical space visualization failed: {e}")


    with tab10:

        st.subheader("t-SNE Chemical Space Visualization")

        st.write(
            "Interactive t-SNE visualization of molecular chemical space using Morgan fingerprints. "
            "Use this tab for single molecule neighborhood search and complete dataset nonlinear clustering."
        )

        try:

            tsne_full_df = load_molecule_dataset()

            tsne_mode = st.radio(
                "Choose t-SNE Mode",
                [
                    "Single Molecule t-SNE Search",
                    "Complete Dataset t-SNE Visualization"
                ],
                horizontal=True,
                key="tsne_visualization_mode"
            )

            selected_tsne_name = None
            selected_tsne_smiles = None

            if tsne_mode == "Single Molecule t-SNE Search":

                st.info(
                    "Search/select one molecule. The selected molecule will be highlighted on the t-SNE plot."
                )

                if "tsne_search_reset_counter" not in st.session_state:
                    st.session_state["tsne_search_reset_counter"] = 0

                tsne_search_key = (
                    "tsne_single_search_"
                    + str(st.session_state["tsne_search_reset_counter"])
                )

                tsne_select_key = (
                    "tsne_single_select_"
                    + str(st.session_state["tsne_search_reset_counter"])
                )

                col_tsne_search, col_tsne_reset = st.columns([5, 1])

                with col_tsne_search:

                    tsne_search_query = st.text_input(
                        "Search molecule by IUPAC/name or SMILES",
                        value="",
                        key=tsne_search_key,
                        placeholder="Example: ethanol, benz, acid, CCO"
                    )

                with col_tsne_reset:

                    st.write("")
                    st.write("")

                    if st.button(
                        "Clear / Reset",
                        key="clear_tsne_single_search"
                    ):

                        for key_to_clear in [
                            "tsne_selected_molecule_name",
                            "tsne_selected_smiles"
                        ]:
                            if key_to_clear in st.session_state:
                                del st.session_state[key_to_clear]

                        st.session_state["tsne_search_reset_counter"] += 1

                        st.rerun()

                if tsne_search_query.strip() != "":

                    tsne_filtered_df = tsne_full_df[
                        tsne_full_df["Molecule_Name"].str.contains(
                            tsne_search_query,
                            case=False,
                            na=False,
                            regex=False
                        )
                        |
                        tsne_full_df["SMILES"].str.contains(
                            tsne_search_query,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ].copy()

                else:

                    tsne_filtered_df = tsne_full_df.copy()

                st.info(
                    f"Matching molecules found: {len(tsne_filtered_df)} "
                    f"out of {len(tsne_full_df)}"
                )

                display_paginated_molecule_table(


                    df=tsne_filtered_df,


                    table_key="tsne_filtered_df_catalog",


                    rows_per_page=100,


                    columns=[


                        "Molecule_Name",


                        "SMILES"


                    ]


                )

                if tsne_filtered_df.empty:

                    st.warning("No molecule found. Please try another search.")

                else:

                    selected_tsne_index = st.selectbox(
                        "Select molecule to highlight on t-SNE plot",
                        options=tsne_filtered_df.index,
                        format_func=lambda x: tsne_filtered_df.loc[
                            x,
                            "Molecule_Display"
                        ],
                        key=tsne_select_key
                    )

                    selected_tsne_name = tsne_filtered_df.loc[
                        selected_tsne_index,
                        "Molecule_Name"
                    ]

                    selected_tsne_smiles = tsne_filtered_df.loc[
                        selected_tsne_index,
                        "SMILES"
                    ]

                    st.session_state["tsne_selected_molecule_name"] = (
                        selected_tsne_name
                    )

                    st.session_state["tsne_selected_smiles"] = (
                        selected_tsne_smiles
                    )

                    st.success(f"Selected Molecule: {selected_tsne_name}")
                    st.success(f"Selected SMILES: {selected_tsne_smiles}")

            st.markdown("---")

            sample_size_tsne = st.slider(
                "Number of molecules to visualize",
                min_value=100,
                max_value=min(3000, len(tsne_full_df)),
                value=min(1000, len(tsne_full_df)),
                step=100,
                key="interactive_tsne_sample_size"
            )

            tsne_perplexity = st.slider(
                "t-SNE perplexity",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                key="interactive_tsne_perplexity"
            )

            tsne_iterations = st.slider(
                "t-SNE iterations",
                min_value=500,
                max_value=2000,
                value=1000,
                step=250,
                key="interactive_tsne_iterations"
            )

            tsne_random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="interactive_tsne_random_seed"
            )

            color_option_tsne = st.selectbox(
                "Color points by",
                [
                    "Point Type",
                    "Scaffold Group",
                    "t-SNE Outlier Status",
                    "t-SNE Chemical Space Status",
                    "Molecule Name"
                ],
                key="interactive_tsne_color_option"
            )

            st.info(
                "Recommended: 500–1000 molecules for faster t-SNE performance. "
                "t-SNE emphasizes local neighborhoods rather than global distances."
            )

            if st.button(
                "Generate Interactive t-SNE Chemical Space Plot",
                key="generate_interactive_tsne_plot"
            ):

                with st.spinner("Generating interactive t-SNE chemical space..."):

                    tsne_df = generate_interactive_tsne_chemical_space(
                        molecule_df=tsne_full_df,
                        selected_smiles=selected_tsne_smiles,
                        sample_size=sample_size_tsne,
                        random_state=int(tsne_random_seed),
                        n_bits=2048,
                        perplexity=tsne_perplexity,
                        max_iter=tsne_iterations
                    )

                if tsne_df.empty:

                    st.warning(
                        "t-SNE visualization could not be generated. Not enough valid molecules."
                    )

                else:

                    st.success("Interactive t-SNE chemical space generated successfully.")

                    col_t1, col_t2, col_t3 = st.columns(3)

                    with col_t1:
                        st.metric(
                            "Molecules Visualized",
                            len(tsne_df)
                        )

                    with col_t2:
                        st.metric(
                            "Perplexity Used",
                            int(tsne_df["tSNE_Perplexity_Used"].iloc[0])
                        )

                    with col_t3:
                        st.metric(
                            "Potential Outliers",
                            int(
                                (
                                    tsne_df["TSNE_Outlier_Status"]
                                    ==
                                    "Potential Outlier"
                                ).sum()
                            )
                        )

                    if color_option_tsne == "Point Type":
                        color_column = "Point_Type"
                    elif color_option_tsne == "Scaffold Group":
                        color_column = "Scaffold_Group"
                    elif color_option_tsne == "t-SNE Outlier Status":
                        color_column = "TSNE_Outlier_Status"
                    elif color_option_tsne == "t-SNE Chemical Space Status":
                        color_column = "TSNE_Chemical_Space_Status"
                    else:
                        color_column = "Molecule_Name"

                    st.subheader("Interactive t-SNE Chemical Space Plot")

                    tsne_fig = px.scatter(
                        tsne_df,
                        x="TSNE_1",
                        y="TSNE_2",
                        color=color_column,
                        hover_data=[
                            "Molecule_Name",
                            "SMILES",
                            "Point_Type",
                            "Murcko_Scaffold",
                            "TSNE_Distance_From_Center",
                            "TSNE_Outlier_Status",
                            "TSNE_Chemical_Space_Status"
                        ],
                        title="Interactive t-SNE Chemical Space Visualization"
                    )

                    tsne_fig.update_traces(
                        marker=dict(
                            size=7,
                            opacity=0.75
                        )
                    )

                    selected_rows = tsne_df[
                        tsne_df["Point_Type"] == "Selected Molecule"
                    ]

                    if not selected_rows.empty:

                        tsne_fig.add_scatter(
                            x=selected_rows["TSNE_1"],
                            y=selected_rows["TSNE_2"],
                            mode="markers+text",
                            text=["Selected Molecule"],
                            textposition="top center",
                            marker=dict(
                                size=18,
                                symbol="star",
                                line=dict(
                                    width=2
                                )
                            ),
                            name="Selected Molecule Highlight"
                        )

                    st.plotly_chart(
                        tsne_fig,
                        width="stretch"
                    )

                    st.markdown("---")

                    if not selected_rows.empty:

                        st.subheader("Selected Molecule t-SNE Location")

                        selected_location_df = selected_rows[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "TSNE_1",
                                "TSNE_2",
                                "TSNE_Distance_From_Center",
                                "TSNE_Outlier_Status",
                                "TSNE_Chemical_Space_Status",
                                "Murcko_Scaffold"
                            ]
                        ].copy()

                        st.dataframe(
                            selected_location_df,
                            width="stretch"
                        )

                        selected_status = selected_location_df[
                            "TSNE_Chemical_Space_Status"
                        ].iloc[0]

                        if selected_status == "Inside t-SNE Neighborhood Space":

                            st.success(
                                "Selected molecule lies inside a t-SNE neighborhood region."
                            )

                        else:

                            st.warning(
                                "Selected molecule appears relatively isolated in t-SNE space."
                            )

                    st.subheader("t-SNE Potential Outlier Table")

                    outlier_df = tsne_df[
                        tsne_df["TSNE_Outlier_Status"] == "Potential Outlier"
                    ].sort_values(
                        by="TSNE_Distance_From_Center",
                        ascending=False
                    )[
                        [
                            "Molecule_Name",
                            "SMILES",
                            "TSNE_1",
                            "TSNE_2",
                            "TSNE_Distance_From_Center",
                            "TSNE_Outlier_Status",
                            "TSNE_Chemical_Space_Status"
                        ]
                    ]

                    if outlier_df.empty:

                        st.info("No t-SNE outliers detected.")

                    else:

                        st.warning(
                            f"{len(outlier_df)} potential t-SNE outlier(s) detected."
                        )

                        st.dataframe(
                            outlier_df.head(50),
                            width="stretch"
                        )

                        outlier_csv = outlier_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download t-SNE Outlier Table CSV",
                            data=outlier_csv,
                            file_name="tsne_outlier_detection_table.csv",
                            mime="text/csv"
                        )

                    st.subheader("Download t-SNE Coordinates")

                    download_cols = [
                        "Molecule_Name",
                        "SMILES",
                        "TSNE_1",
                        "TSNE_2",
                        "Point_Type",
                        "Murcko_Scaffold",
                        "Scaffold_Group",
                        "TSNE_Distance_From_Center",
                        "TSNE_Outlier_Status",
                        "TSNE_Chemical_Space_Status",
                        "tSNE_Perplexity_Used"
                    ]

                    tsne_csv = tsne_df[download_cols].to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download t-SNE Coordinates CSV",
                        data=tsne_csv,
                        file_name="interactive_tsne_chemical_space_coordinates.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Interpretation: t-SNE is best for local molecular neighborhood "
                        "patterns. Close points are likely fingerprint-similar, while isolated "
                        "points may represent unusual chemistry."
                    )

        except Exception as e:

            st.error(f"t-SNE chemical space visualization failed: {e}")


    with tab11:

        st.subheader("UMAP Chemical Space Visualization")

        st.write(
            "Interactive UMAP visualization of molecular chemical space using Morgan fingerprints. "
            "Use this tab for single molecule neighborhood search and complete dataset nonlinear clustering."
        )

        if not UMAP_AVAILABLE:

            st.warning(
                "UMAP is not installed. Please install it first:"
            )

            st.code(
                "pip install umap-learn",
                language="bash"
            )

        else:

            try:

                umap_full_df = load_molecule_dataset()

                umap_mode = st.radio(
                    "Choose UMAP Mode",
                    [
                        "Single Molecule UMAP Search",
                        "Complete Dataset UMAP Visualization"
                    ],
                    horizontal=True,
                    key="umap_visualization_mode"
                )

                selected_umap_name = None
                selected_umap_smiles = None

                if umap_mode == "Single Molecule UMAP Search":

                    st.info(
                        "Search/select one molecule. The selected molecule will be highlighted on the UMAP plot."
                    )

                    if "umap_search_reset_counter" not in st.session_state:
                        st.session_state["umap_search_reset_counter"] = 0

                    umap_search_key = (
                        "umap_single_search_"
                        + str(st.session_state["umap_search_reset_counter"])
                    )

                    umap_select_key = (
                        "umap_single_select_"
                        + str(st.session_state["umap_search_reset_counter"])
                    )

                    col_umap_search, col_umap_reset = st.columns([5, 1])

                    with col_umap_search:

                        umap_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=umap_search_key,
                            placeholder="Example: ethanol, benz, acid, CCO"
                        )

                    with col_umap_reset:

                        st.write("")
                        st.write("")

                        if st.button(
                            "Clear / Reset",
                            key="clear_umap_single_search"
                        ):

                            for key_to_clear in [
                                "umap_selected_molecule_name",
                                "umap_selected_smiles"
                            ]:
                                if key_to_clear in st.session_state:
                                    del st.session_state[key_to_clear]

                            st.session_state["umap_search_reset_counter"] += 1

                            st.rerun()

                    if umap_search_query.strip() != "":

                        umap_filtered_df = umap_full_df[
                            umap_full_df["Molecule_Name"].str.contains(
                                umap_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                            |
                            umap_full_df["SMILES"].str.contains(
                                umap_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                        ].copy()

                    else:

                        umap_filtered_df = umap_full_df.copy()

                    st.info(
                        f"Matching molecules found: {len(umap_filtered_df)} "
                        f"out of {len(umap_full_df)}"
                    )

                    display_paginated_molecule_table(


                        df=umap_filtered_df,


                        table_key="umap_filtered_df_catalog",


                        rows_per_page=100,


                        columns=[


                            "Molecule_Name",


                            "SMILES"


                        ]


                    )

                    if umap_filtered_df.empty:

                        st.warning("No molecule found. Please try another search.")

                    else:

                        selected_umap_index = st.selectbox(
                            "Select molecule to highlight on UMAP plot",
                            options=umap_filtered_df.index,
                            format_func=lambda x: umap_filtered_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=umap_select_key
                        )

                        selected_umap_name = umap_filtered_df.loc[
                            selected_umap_index,
                            "Molecule_Name"
                        ]

                        selected_umap_smiles = umap_filtered_df.loc[
                            selected_umap_index,
                            "SMILES"
                        ]

                        st.session_state["umap_selected_molecule_name"] = (
                            selected_umap_name
                        )

                        st.session_state["umap_selected_smiles"] = (
                            selected_umap_smiles
                        )

                        st.success(f"Selected Molecule: {selected_umap_name}")
                        st.success(f"Selected SMILES: {selected_umap_smiles}")

                st.markdown("---")

                sample_size_umap = st.slider(
                    "Number of molecules to visualize",
                    min_value=100,
                    max_value=min(3000, len(umap_full_df)),
                    value=min(1500, len(umap_full_df)),
                    step=100,
                    key="interactive_umap_sample_size"
                )

                umap_n_neighbors = st.slider(
                    "UMAP n_neighbors",
                    min_value=5,
                    max_value=100,
                    value=15,
                    step=5,
                    key="interactive_umap_n_neighbors"
                )

                umap_min_dist = st.slider(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.10,
                    step=0.01,
                    key="interactive_umap_min_dist"
                )

                umap_random_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    key="interactive_umap_random_seed"
                )

                color_option_umap = st.selectbox(
                    "Color points by",
                    [
                        "Point Type",
                        "Scaffold Group",
                        "UMAP Outlier Status",
                        "UMAP Chemical Space Status",
                        "Molecule Name"
                    ],
                    key="interactive_umap_color_option"
                )

                st.info(
                    "Recommended: 1000–2000 molecules. UMAP usually gives the clearest "
                    "chemical-space cluster visualization."
                )

                if st.button(
                    "Generate Interactive UMAP Chemical Space Plot",
                    key="generate_interactive_umap_plot"
                ):

                    with st.spinner("Generating interactive UMAP chemical space..."):

                        umap_df = generate_interactive_umap_chemical_space(
                            molecule_df=umap_full_df,
                            selected_smiles=selected_umap_smiles,
                            sample_size=sample_size_umap,
                            random_state=int(umap_random_seed),
                            n_bits=2048,
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist
                        )

                    if umap_df.empty:

                        st.warning(
                            "UMAP visualization could not be generated. Not enough valid molecules."
                        )

                    else:

                        st.success("Interactive UMAP chemical space generated successfully.")

                        col_u1, col_u2, col_u3, col_u4 = st.columns(4)

                        with col_u1:
                            st.metric(
                                "Molecules Visualized",
                                len(umap_df)
                            )

                        with col_u2:
                            st.metric(
                                "n_neighbors",
                                int(umap_df["UMAP_n_neighbors_Used"].iloc[0])
                            )

                        with col_u3:
                            st.metric(
                                "min_dist",
                                f"{float(umap_df['UMAP_min_dist_Used'].iloc[0]):.2f}"
                            )

                        with col_u4:
                            st.metric(
                                "Potential Outliers",
                                int(
                                    (
                                        umap_df["UMAP_Outlier_Status"]
                                        ==
                                        "Potential Outlier"
                                    ).sum()
                                )
                            )

                        if color_option_umap == "Point Type":
                            color_column = "Point_Type"
                        elif color_option_umap == "Scaffold Group":
                            color_column = "Scaffold_Group"
                        elif color_option_umap == "UMAP Outlier Status":
                            color_column = "UMAP_Outlier_Status"
                        elif color_option_umap == "UMAP Chemical Space Status":
                            color_column = "UMAP_Chemical_Space_Status"
                        else:
                            color_column = "Molecule_Name"

                        st.subheader("Interactive UMAP Chemical Space Plot")

                        umap_fig = px.scatter(
                            umap_df,
                            x="UMAP_1",
                            y="UMAP_2",
                            color=color_column,
                            hover_data=[
                                "Molecule_Name",
                                "SMILES",
                                "Point_Type",
                                "Murcko_Scaffold",
                                "UMAP_Distance_From_Center",
                                "UMAP_Outlier_Status",
                                "UMAP_Chemical_Space_Status"
                            ],
                            title="Interactive UMAP Chemical Space Visualization"
                        )

                        umap_fig.update_traces(
                            marker=dict(
                                size=7,
                                opacity=0.75
                            )
                        )

                        selected_rows = umap_df[
                            umap_df["Point_Type"] == "Selected Molecule"
                        ]

                        if not selected_rows.empty:

                            umap_fig.add_scatter(
                                x=selected_rows["UMAP_1"],
                                y=selected_rows["UMAP_2"],
                                mode="markers+text",
                                text=["Selected Molecule"],
                                textposition="top center",
                                marker=dict(
                                    size=18,
                                    symbol="star",
                                    line=dict(
                                        width=2
                                    )
                                ),
                                name="Selected Molecule Highlight"
                            )

                        st.plotly_chart(
                            umap_fig,
                            width="stretch"
                        )

                        st.markdown("---")

                        if not selected_rows.empty:

                            st.subheader("Selected Molecule UMAP Location")

                            selected_location_df = selected_rows[
                                [
                                    "Molecule_Name",
                                    "SMILES",
                                    "UMAP_1",
                                    "UMAP_2",
                                    "UMAP_Distance_From_Center",
                                    "UMAP_Outlier_Status",
                                    "UMAP_Chemical_Space_Status",
                                    "Murcko_Scaffold"
                                ]
                            ].copy()

                            st.dataframe(
                                selected_location_df,
                                width="stretch"
                            )

                            selected_status = selected_location_df[
                                "UMAP_Chemical_Space_Status"
                            ].iloc[0]

                            if selected_status == "Inside UMAP Neighborhood Space":

                                st.success(
                                    "Selected molecule lies inside a UMAP neighborhood region."
                                )

                            else:

                                st.warning(
                                    "Selected molecule appears relatively isolated in UMAP space."
                                )

                        st.subheader("UMAP Potential Outlier Table")

                        outlier_df = umap_df[
                            umap_df["UMAP_Outlier_Status"] == "Potential Outlier"
                        ].sort_values(
                            by="UMAP_Distance_From_Center",
                            ascending=False
                        )[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "UMAP_1",
                                "UMAP_2",
                                "UMAP_Distance_From_Center",
                                "UMAP_Outlier_Status",
                                "UMAP_Chemical_Space_Status"
                            ]
                        ]

                        if outlier_df.empty:

                            st.info("No UMAP outliers detected.")

                        else:

                            st.warning(
                                f"{len(outlier_df)} potential UMAP outlier(s) detected."
                            )

                            st.dataframe(
                                outlier_df.head(50),
                                width="stretch"
                            )

                            outlier_csv = outlier_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download UMAP Outlier Table CSV",
                                data=outlier_csv,
                                file_name="umap_outlier_detection_table.csv",
                                mime="text/csv"
                            )

                        st.subheader("Download UMAP Coordinates")

                        download_cols = [
                            "Molecule_Name",
                            "SMILES",
                            "UMAP_1",
                            "UMAP_2",
                            "Point_Type",
                            "Murcko_Scaffold",
                            "Scaffold_Group",
                            "UMAP_Distance_From_Center",
                            "UMAP_Outlier_Status",
                            "UMAP_Chemical_Space_Status",
                            "UMAP_n_neighbors_Used",
                            "UMAP_min_dist_Used"
                        ]

                        umap_csv = umap_df[download_cols].to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download UMAP Coordinates CSV",
                            data=umap_csv,
                            file_name="interactive_umap_chemical_space_coordinates.csv",
                            mime="text/csv"
                        )

                        st.info(
                            "Interpretation: UMAP is useful for nonlinear chemical-space "
                            "neighborhoods and scaffold-like clustering. Isolated regions may "
                            "indicate unusual chemistry or potential OOD behavior."
                        )

            except Exception as e:

                st.error(f"UMAP chemical space visualization failed: {e}")


    with tab12:

        st.subheader("Interactive Plotly UMAP Visualization")

        st.write(
            "Advanced interactive UMAP chemical-space explorer using Plotly. "
            "Use this tab for single molecule highlighting, full dataset exploration, "
            "AI overlay visualization, scaffold grouping, outlier analysis, and export."
        )

        if not UMAP_AVAILABLE:

            st.warning(
                "UMAP is not installed. Please install it first:"
            )

            st.code(
                "pip install umap-learn",
                language="bash"
            )

        else:

            try:

                plotly_umap_full_df = load_molecule_dataset()

                plotly_umap_mode = st.radio(
                    "Choose Interactive UMAP Mode",
                    [
                        "Single Molecule Interactive UMAP Search",
                        "Complete Dataset Interactive UMAP"
                    ],
                    horizontal=True,
                    key="plotly_umap_mode_updated"
                )

                selected_plotly_umap_name = None
                selected_plotly_umap_smiles = None

                # ==================================================
                # SINGLE MOLECULE SEARCH MODE
                # ==================================================

                if plotly_umap_mode == "Single Molecule Interactive UMAP Search":

                    st.info(
                        "Search/select one molecule. The selected molecule will be highlighted "
                        "as a star on the interactive Plotly UMAP plot."
                    )

                    if "plotly_umap_search_reset_counter" not in st.session_state:
                        st.session_state["plotly_umap_search_reset_counter"] = 0

                    plotly_umap_search_key = (
                        "plotly_umap_single_search_"
                        + str(st.session_state["plotly_umap_search_reset_counter"])
                    )

                    plotly_umap_select_key = (
                        "plotly_umap_single_select_"
                        + str(st.session_state["plotly_umap_search_reset_counter"])
                    )

                    col_pu_search, col_pu_reset = st.columns([5, 1])

                    with col_pu_search:

                        plotly_umap_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=plotly_umap_search_key,
                            placeholder="Example: ethanol, benz, acid, CCO"
                        )

                    with col_pu_reset:

                        st.write("")
                        st.write("")

                        if st.button(
                            "Clear / Reset",
                            key="clear_plotly_umap_single_search"
                        ):

                            for key_to_clear in [
                                "plotly_umap_selected_molecule_name",
                                "plotly_umap_selected_smiles"
                            ]:
                                if key_to_clear in st.session_state:
                                    del st.session_state[key_to_clear]

                            st.session_state["plotly_umap_search_reset_counter"] += 1

                            st.rerun()

                    if plotly_umap_search_query.strip() != "":

                        plotly_umap_filtered_df = plotly_umap_full_df[
                            plotly_umap_full_df["Molecule_Name"].str.contains(
                                plotly_umap_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                            |
                            plotly_umap_full_df["SMILES"].str.contains(
                                plotly_umap_search_query,
                                case=False,
                                na=False,
                                regex=False
                            )
                        ].copy()

                    else:

                        plotly_umap_filtered_df = plotly_umap_full_df.copy()

                    st.info(
                        f"Matching molecules found: {len(plotly_umap_filtered_df)} "
                        f"out of {len(plotly_umap_full_df)}"
                    )

                    display_paginated_molecule_table(


                        df=plotly_umap_filtered_df,


                        table_key="plotly_umap_filtered_df_catalog",


                        rows_per_page=100,


                        columns=[


                            "Molecule_Name",


                            "SMILES"


                        ]


                    )

                    if plotly_umap_filtered_df.empty:

                        st.warning("No molecule found. Please try another search.")

                    else:

                        selected_plotly_umap_index = st.selectbox(
                            "Select molecule to highlight on interactive UMAP plot",
                            options=plotly_umap_filtered_df.index,
                            format_func=lambda x: plotly_umap_filtered_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=plotly_umap_select_key
                        )

                        selected_plotly_umap_name = plotly_umap_filtered_df.loc[
                            selected_plotly_umap_index,
                            "Molecule_Name"
                        ]

                        selected_plotly_umap_smiles = plotly_umap_filtered_df.loc[
                            selected_plotly_umap_index,
                            "SMILES"
                        ]

                        st.session_state["plotly_umap_selected_molecule_name"] = (
                            selected_plotly_umap_name
                        )

                        st.session_state["plotly_umap_selected_smiles"] = (
                            selected_plotly_umap_smiles
                        )

                        st.success(
                            f"Selected Molecule: {selected_plotly_umap_name}"
                        )

                        st.success(
                            f"Selected SMILES: {selected_plotly_umap_smiles}"
                        )

                st.markdown("---")

                sample_size_plotly_umap = st.slider(
                    "Number of molecules for interactive Plotly UMAP",
                    min_value=100,
                    max_value=min(3000, len(plotly_umap_full_df)),
                    value=min(1500, len(plotly_umap_full_df)),
                    step=100,
                    key="plotly_umap_updated_sample_size"
                )

                plotly_umap_n_neighbors = st.slider(
                    "UMAP n_neighbors",
                    min_value=5,
                    max_value=100,
                    value=15,
                    step=5,
                    key="plotly_umap_updated_n_neighbors"
                )

                plotly_umap_min_dist = st.slider(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.10,
                    step=0.01,
                    key="plotly_umap_updated_min_dist"
                )

                plotly_umap_random_state = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    key="plotly_umap_updated_random_state"
                )

                color_option_plotly_umap = st.selectbox(
                    "Color points by",
                    [
                        "Point Type",
                        "Scaffold Group",
                        "UMAP Outlier Status",
                        "UMAP Chemical Space Status",
                        "Predicted Melting Point",
                        "Confidence %",
                        "OOD Status",
                        "Molecule Name"
                    ],
                    key="plotly_umap_updated_color_option"
                )

                add_ai_overlay = color_option_plotly_umap in [
                    "Predicted Melting Point",
                    "Confidence %",
                    "OOD Status"
                ]

                if add_ai_overlay:

                    st.warning(
                        "AI overlay mode calculates predictions/OOD for each plotted molecule. "
                        "Use 100–500 molecules for faster performance."
                    )

                st.info(
                    "Recommended: 1000–2000 molecules for standard visualization, "
                    "100–500 molecules when using AI overlay coloring."
                )

                if st.button(
                    "Generate Updated Interactive Plotly UMAP",
                    key="generate_updated_interactive_plotly_umap"
                ):

                    with st.spinner("Generating updated interactive Plotly UMAP..."):

                        plotly_umap_df = generate_interactive_umap_chemical_space(
                            molecule_df=plotly_umap_full_df,
                            selected_smiles=selected_plotly_umap_smiles,
                            sample_size=sample_size_plotly_umap,
                            random_state=int(plotly_umap_random_state),
                            n_bits=2048,
                            n_neighbors=plotly_umap_n_neighbors,
                            min_dist=plotly_umap_min_dist
                        )

                        if not plotly_umap_df.empty and add_ai_overlay:

                            ai_overlay_rows = []

                            reference_df_for_ood = load_molecule_dataset()

                            for _, overlay_row in plotly_umap_df.iterrows():

                                overlay_smiles = overlay_row["SMILES"]

                                try:

                                    overlay_rdkit_pred = float(
                                        predict_melting_point(overlay_smiles)
                                    )

                                    overlay_hybrid_pred = float(
                                        predict_hybrid_gat(overlay_smiles)
                                    )

                                    overlay_ensemble_pred = (
                                        0.4 * overlay_rdkit_pred
                                        +
                                        0.6 * overlay_hybrid_pred
                                    )

                                    overlay_uncertainty = calculate_prediction_uncertainty(
                                        overlay_rdkit_pred,
                                        overlay_hybrid_pred
                                    )

                                    overlay_ood_result = detect_ood_molecule(
                                        overlay_smiles,
                                        reference_df_for_ood
                                    )

                                    ai_overlay_rows.append({
                                        "SMILES": overlay_smiles,
                                        "Predicted_Melting_Point_K": round(
                                            overlay_ensemble_pred,
                                            2
                                        ),
                                        "Predicted_Melting_Point_C": round(
                                            overlay_ensemble_pred - 273.15,
                                            2
                                        ),
                                        "Confidence_%": overlay_uncertainty[
                                            "confidence"
                                        ],
                                        "Model_Difference_K": overlay_uncertainty[
                                            "difference"
                                        ],
                                        "Estimated_Uncertainty_K": overlay_uncertainty[
                                            "uncertainty_range"
                                        ],
                                        "OOD_Status": overlay_ood_result[
                                            "OOD_Status"
                                        ],
                                        "Nearest_Similarity": overlay_ood_result[
                                            "Max_Tanimoto_Similarity"
                                        ]
                                    })

                                except Exception:

                                    ai_overlay_rows.append({
                                        "SMILES": overlay_smiles,
                                        "Predicted_Melting_Point_K": None,
                                        "Predicted_Melting_Point_C": None,
                                        "Confidence_%": None,
                                        "Model_Difference_K": None,
                                        "Estimated_Uncertainty_K": None,
                                        "OOD_Status": "Prediction Failed",
                                        "Nearest_Similarity": None
                                    })

                            ai_overlay_df = pd.DataFrame(ai_overlay_rows)

                            plotly_umap_df = plotly_umap_df.merge(
                                ai_overlay_df,
                                on="SMILES",
                                how="left"
                            )

                    if plotly_umap_df.empty:

                        st.warning(
                            "Interactive Plotly UMAP could not be generated. "
                            "Not enough valid molecules."
                        )

                    else:

                        st.success(
                            "Updated Interactive Plotly UMAP generated successfully."
                        )

                        col_pu1, col_pu2, col_pu3, col_pu4 = st.columns(4)

                        with col_pu1:
                            st.metric(
                                "Molecules Visualized",
                                len(plotly_umap_df)
                            )

                        with col_pu2:
                            st.metric(
                                "n_neighbors",
                                int(plotly_umap_df["UMAP_n_neighbors_Used"].iloc[0])
                            )

                        with col_pu3:
                            st.metric(
                                "min_dist",
                                f"{float(plotly_umap_df['UMAP_min_dist_Used'].iloc[0]):.2f}"
                            )

                        with col_pu4:
                            st.metric(
                                "Potential Outliers",
                                int(
                                    (
                                        plotly_umap_df["UMAP_Outlier_Status"]
                                        ==
                                        "Potential Outlier"
                                    ).sum()
                                )
                            )

                        if color_option_plotly_umap == "Point Type":
                            color_column = "Point_Type"
                        elif color_option_plotly_umap == "Scaffold Group":
                            color_column = "Scaffold_Group"
                        elif color_option_plotly_umap == "UMAP Outlier Status":
                            color_column = "UMAP_Outlier_Status"
                        elif color_option_plotly_umap == "UMAP Chemical Space Status":
                            color_column = "UMAP_Chemical_Space_Status"
                        elif color_option_plotly_umap == "Predicted Melting Point":
                            color_column = "Predicted_Melting_Point_K"
                        elif color_option_plotly_umap == "Confidence %":
                            color_column = "Confidence_%"
                        elif color_option_plotly_umap == "OOD Status":
                            color_column = "OOD_Status"
                        else:
                            color_column = "Molecule_Name"

                        hover_cols = [
                            "Molecule_Name",
                            "SMILES",
                            "Point_Type",
                            "Murcko_Scaffold",
                            "Scaffold_Group",
                            "UMAP_Distance_From_Center",
                            "UMAP_Outlier_Status",
                            "UMAP_Chemical_Space_Status"
                        ]

                        optional_hover_cols = [
                            "Predicted_Melting_Point_K",
                            "Predicted_Melting_Point_C",
                            "Confidence_%",
                            "Model_Difference_K",
                            "Estimated_Uncertainty_K",
                            "OOD_Status",
                            "Nearest_Similarity"
                        ]

                        for optional_col in optional_hover_cols:
                            if optional_col in plotly_umap_df.columns:
                                hover_cols.append(optional_col)

                        st.subheader("Updated Interactive Plotly UMAP Plot")

                        plotly_fig = px.scatter(
                            plotly_umap_df,
                            x="UMAP_1",
                            y="UMAP_2",
                            color=color_column,
                            hover_data=hover_cols,
                            title=f"Interactive Plotly UMAP Colored by {color_option_plotly_umap}",
                            color_continuous_scale=(
                                "Viridis"
                                if color_option_plotly_umap in [
                                    "Predicted Melting Point",
                                    "Confidence %"
                                ]
                                else None
                            )
                        )

                        plotly_fig.update_traces(
                            marker=dict(
                                size=7,
                                opacity=0.75
                            )
                        )

                        selected_rows = plotly_umap_df[
                            plotly_umap_df["Point_Type"] == "Selected Molecule"
                        ]

                        if not selected_rows.empty:

                            plotly_fig.add_scatter(
                                x=selected_rows["UMAP_1"],
                                y=selected_rows["UMAP_2"],
                                mode="markers+text",
                                text=["Selected Molecule"],
                                textposition="top center",
                                marker=dict(
                                    size=20,
                                    symbol="star",
                                    line=dict(
                                        width=2
                                    )
                                ),
                                name="Selected Molecule Highlight"
                            )

                        plotly_fig.update_layout(
                            height=700,
                            hovermode="closest"
                        )

                        st.plotly_chart(
                            plotly_fig,
                            width="stretch"
                        )

                        st.markdown("---")

                        if not selected_rows.empty:

                            st.subheader("Selected Molecule UMAP Location")

                            selected_location_cols = [
                                "Molecule_Name",
                                "SMILES",
                                "UMAP_1",
                                "UMAP_2",
                                "UMAP_Distance_From_Center",
                                "UMAP_Outlier_Status",
                                "UMAP_Chemical_Space_Status",
                                "Murcko_Scaffold"
                            ]

                            extra_cols = [
                                "Predicted_Melting_Point_K",
                                "Confidence_%",
                                "OOD_Status",
                                "Nearest_Similarity"
                            ]

                            for extra_col in extra_cols:
                                if extra_col in plotly_umap_df.columns:
                                    selected_location_cols.append(extra_col)

                            selected_location_df = selected_rows[
                                selected_location_cols
                            ].copy()

                            st.dataframe(
                                selected_location_df,
                                width="stretch"
                            )

                            selected_status = selected_location_df[
                                "UMAP_Chemical_Space_Status"
                            ].iloc[0]

                            if selected_status == "Inside UMAP Neighborhood Space":

                                st.success(
                                    "Selected molecule lies inside a UMAP neighborhood region."
                                )

                            else:

                                st.warning(
                                    "Selected molecule appears relatively isolated in UMAP space."
                                )

                        st.subheader("UMAP Potential Outlier Table")

                        outlier_df = plotly_umap_df[
                            plotly_umap_df["UMAP_Outlier_Status"] == "Potential Outlier"
                        ].sort_values(
                            by="UMAP_Distance_From_Center",
                            ascending=False
                        )

                        if outlier_df.empty:

                            st.info("No UMAP outliers detected.")

                        else:

                            st.warning(
                                f"{len(outlier_df)} potential UMAP outlier(s) detected."
                            )

                            outlier_display_cols = [
                                "Molecule_Name",
                                "SMILES",
                                "UMAP_1",
                                "UMAP_2",
                                "UMAP_Distance_From_Center",
                                "UMAP_Outlier_Status",
                                "UMAP_Chemical_Space_Status"
                            ]

                            st.dataframe(
                                outlier_df[outlier_display_cols].head(50),
                                width="stretch"
                            )

                            outlier_csv = outlier_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download UMAP Outlier Table CSV",
                                data=outlier_csv,
                                file_name="interactive_plotly_umap_outliers.csv",
                                mime="text/csv"
                            )

                        st.subheader("Download Interactive UMAP Data")

                        umap_csv = plotly_umap_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Interactive UMAP Coordinates CSV",
                            data=umap_csv,
                            file_name="interactive_plotly_umap_coordinates.csv",
                            mime="text/csv"
                        )

                        html_content = plotly_fig.to_html(
                            include_plotlyjs="cdn"
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Interactive UMAP Plot HTML",
                            data=html_content,
                            file_name="interactive_plotly_umap_plot.html",
                            mime="text/html"
                        )

                        st.info(
                            "Interpretation: Interactive UMAP supports zooming, panning, hovering, "
                            "selected molecule highlighting, scaffold exploration, outlier review, "
                            "and optional AI overlay coloring."
                        )

            except Exception as e:

                st.error(f"Interactive Plotly UMAP visualization failed: {e}")



    # ==================================================
    # TAB 13 — DRUG-LIKENESS ANALYSIS
    # ==================================================

    with tab13:

        st.subheader("Drug-Likeness Analysis")

        st.write(
            "Analyze molecular drug-likeness using RDKit descriptors, Lipinski Rule of 5, "
            "Veber Rule, Ghose Filter, lead-likeness, PAINS alerts, synthetic accessibility estimate, "
            "batch analysis, radar chart, and PDF reporting."
        )

        try:

            drug_df = load_molecule_dataset()

            drug_analysis_mode = st.radio(
                "Choose Drug-Likeness Analysis Mode",
                [
                    "Single Molecule Drug-Likeness",
                    "Batch Drug-Likeness CSV Analysis"
                ],
                horizontal=True,
                key="drug_likeness_analysis_mode"
            )

            # ==================================================
            # SINGLE MOLECULE DRUG-LIKENESS
            # ==================================================

            if drug_analysis_mode == "Single Molecule Drug-Likeness":

                drug_input_mode = st.radio(
                    "Choose Drug-Likeness Input Method",
                    [
                        "Select from Dataset",
                        "Enter Custom SMILES"
                    ],
                    horizontal=True,
                    key="drug_likeness_input_mode"
                )

                drug_name = "Custom Input"
                drug_smiles = "CCO"

                if drug_input_mode == "Select from Dataset":

                    st.info(
                        "Search or browse the available molecule catalog, then select a molecule "
                        "for drug-likeness analysis."
                    )

                    drug_catalog_csv = drug_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Full Molecule Catalog CSV",
                        data=drug_catalog_csv,
                        file_name="drug_likeness_full_molecule_catalog.csv",
                        mime="text/csv",
                        key="download_drug_likeness_full_catalog"
                    )

                    if "drug_search_reset_counter" not in st.session_state:
                        st.session_state["drug_search_reset_counter"] = 0

                    drug_search_key = (
                        "drug_likeness_search_"
                        + str(st.session_state["drug_search_reset_counter"])
                    )

                    drug_selectbox_key = (
                        "drug_likeness_selectbox_"
                        + str(st.session_state["drug_search_reset_counter"])
                    )

                    with st.expander(
                        "Search Available Molecule Catalog for Drug-Likeness Analysis",
                        expanded=True
                    ):

                        col_drug_search, col_drug_reset = st.columns([5, 1])

                        with col_drug_search:

                            drug_search_query = st.text_input(
                                "Search molecule by IUPAC/name or SMILES",
                                value="",
                                key=drug_search_key,
                                placeholder="Example: ethanol, benz, acid, CCO"
                            )

                        with col_drug_reset:

                            st.write("")
                            st.write("")

                            if st.button(
                                "Clear / Reset",
                                key="clear_drug_likeness_search"
                            ):

                                for key_to_clear in [
                                    "drug_selected_molecule_name",
                                    "drug_selected_smiles"
                                ]:
                                    if key_to_clear in st.session_state:
                                        del st.session_state[key_to_clear]

                                st.session_state["drug_search_reset_counter"] += 1
                                st.rerun()

                        if drug_search_query.strip() != "":

                            drug_filtered_df = drug_df[
                                drug_df["Molecule_Name"].str.contains(
                                    drug_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                                |
                                drug_df["SMILES"].str.contains(
                                    drug_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                            ].copy()

                        else:

                            drug_filtered_df = drug_df.copy()

                        st.info(
                            f"Matching molecules found: {len(drug_filtered_df)} "
                            f"out of {len(drug_df)}"
                        )

                        display_paginated_molecule_table(
                            df=drug_filtered_df,
                            table_key="drug_filtered_df_catalog",
                            rows_per_page=100,
                            columns=[
                                "Molecule_Name",
                                "SMILES"
                            ]
                        )

                        filtered_drug_csv = drug_filtered_df[
                            [
                                "Molecule_Name",
                                "SMILES"
                            ]
                        ].to_csv(index=False).encode("utf-8")

                        st.download_button(
                            label="Download Filtered Molecule List CSV",
                            data=filtered_drug_csv,
                            file_name="drug_likeness_filtered_molecule_catalog.csv",
                            mime="text/csv",
                            key="download_drug_likeness_filtered_catalog"
                        )

                        st.markdown("---")

                        if drug_filtered_df.empty:

                            st.warning(
                                "No molecule found for this search. Please try another molecule name or SMILES."
                            )

                        else:

                            st.subheader("Select Molecule for Drug-Likeness Analysis")

                            selected_drug_index = st.selectbox(
                                "Choose molecule from filtered list",
                                options=drug_filtered_df.index,
                                format_func=lambda x: drug_filtered_df.loc[
                                    x,
                                    "Molecule_Display"
                                ],
                                key=drug_selectbox_key
                            )

                            drug_name = drug_filtered_df.loc[
                                selected_drug_index,
                                "Molecule_Name"
                            ]

                            drug_smiles = drug_filtered_df.loc[
                                selected_drug_index,
                                "SMILES"
                            ]

                            st.session_state["drug_selected_molecule_name"] = (
                                drug_name
                            )

                            st.session_state["drug_selected_smiles"] = (
                                drug_smiles
                            )

                    if "drug_filtered_df" in locals() and not drug_filtered_df.empty:

                        drug_name = st.session_state.get(
                            "drug_selected_molecule_name",
                            drug_filtered_df.iloc[0]["Molecule_Name"]
                        )

                        drug_smiles = st.session_state.get(
                            "drug_selected_smiles",
                            drug_filtered_df.iloc[0]["SMILES"]
                        )

                    else:

                        drug_name = "Dataset Selection"
                        drug_smiles = "CCO"

                else:

                    drug_name = st.text_input(
                        "Molecule Name / Label",
                        value="Custom Input",
                        key="drug_custom_name"
                    )

                    drug_smiles = st.text_input(
                        "Enter Custom SMILES",
                        value="CCO",
                        key="drug_custom_smiles"
                    )

                st.subheader("Current Molecule Selected for Drug-Likeness Analysis")

                st.success(f"Molecule: {drug_name}")
                st.success(f"SMILES: {drug_smiles}")

                st.text_area(
                    "Copy Molecule Name",
                    value=drug_name,
                    height=70,
                    key=f"drug_copy_name_{make_safe_filename(drug_name)}"
                )

                st.text_area(
                    "Copy SMILES",
                    value=drug_smiles,
                    height=70,
                    key=f"drug_copy_smiles_{make_safe_filename(drug_smiles)}"
                )

                if st.button(
                    "Run Drug-Likeness Analysis",
                    key="run_drug_likeness_analysis"
                ):

                    mol = Chem.MolFromSmiles(drug_smiles)

                    if mol is None:

                        st.error(
                            "Invalid SMILES. Please enter or select a valid molecule."
                        )

                    else:

                        report_df = generate_drug_likeness_report_dataframe(
                            molecule_name=drug_name,
                            smiles=drug_smiles
                        )

                        if report_df.empty:

                            st.error(
                                "Drug-likeness properties could not be calculated."
                            )

                        else:

                            drug_properties = report_df.iloc[0].to_dict()

                            st.subheader("Molecular Structure")

                            drug_image = Draw.MolToImage(
                                mol,
                                size=(350, 350)
                            )

                            st.image(
                                drug_image,
                                caption="Selected Molecule 2D Structure"
                            )

                            st.markdown("---")

                            st.subheader("Drug-Likeness Summary")

                            col_d1, col_d2, col_d3, col_d4 = st.columns(4)

                            with col_d1:
                                st.metric(
                                    "Lipinski",
                                    drug_properties["Lipinski_Status"]
                                )

                            with col_d2:
                                st.metric(
                                    "Veber",
                                    drug_properties["Veber_Status"]
                                )

                            with col_d3:
                                st.metric(
                                    "Ghose",
                                    drug_properties["Ghose_Status"]
                                )

                            with col_d4:
                                st.metric(
                                    "Lead-Likeness",
                                    drug_properties["Lead_Likeness_Status"]
                                )

                            col_d5, col_d6, col_d7, col_d8 = st.columns(4)

                            with col_d5:
                                st.metric(
                                    "PAINS",
                                    drug_properties["PAINS_Status"]
                                )

                            with col_d6:
                                st.metric(
                                    "SA Estimate",
                                    drug_properties["Synthetic_Accessibility_Estimate"]
                                )

                            with col_d7:
                                st.metric(
                                    "Bioavailability",
                                    f"{drug_properties['Bioavailability_Score_%']}%"
                                )

                            with col_d8:
                                st.metric(
                                    "Drug-Likeness",
                                    drug_properties["Drug_Likeness_Label"]
                                )

                            if drug_properties["Lipinski_Status"] == "Pass":
                                st.success(
                                    "Lipinski Rule of 5: PASS — molecule satisfies all core drug-likeness rules."
                                )
                            elif drug_properties["Lipinski_Status"] == "Acceptable":
                                st.warning(
                                    "Lipinski Rule of 5: ACCEPTABLE — molecule has one violation and may still be useful."
                                )
                            else:
                                st.error(
                                    "Lipinski Rule of 5: FAIL — molecule has multiple violations."
                                )

                            if drug_properties["PAINS_Status"] == "No PAINS Alerts":
                                st.success("PAINS screening: no PAINS alerts detected.")
                            elif drug_properties["PAINS_Status"] == "PAINS Screening Unavailable":
                                st.warning("PAINS screening unavailable in current RDKit environment.")
                            else:
                                st.error(
                                    f"PAINS screening: {drug_properties['PAINS_Alert_Count']} alert(s) detected."
                                )

                            st.info(
                                drug_properties["Bioavailability_Label"]
                            )

                            st.markdown("---")

                            st.subheader("Drug-Likeness Radar Chart")

                            radar_metrics = {
                                "MW": max(
                                    0,
                                    min(
                                        100,
                                        100 - abs(drug_properties["Molecular_Weight"] - 350) / 350 * 100
                                    )
                                ),
                                "LogP": max(
                                    0,
                                    min(
                                        100,
                                        100 - abs(drug_properties["LogP"] - 2.5) / 5 * 100
                                    )
                                ),
                                "TPSA": max(
                                    0,
                                    min(
                                        100,
                                        100 - drug_properties["TPSA"] / 140 * 100
                                    )
                                ),
                                "RotB": max(
                                    0,
                                    min(
                                        100,
                                        100 - drug_properties["Rotatable_Bonds"] / 10 * 100
                                    )
                                ),
                                "HBD": max(
                                    0,
                                    min(
                                        100,
                                        100 - drug_properties["H_Bond_Donors"] / 5 * 100
                                    )
                                ),
                                "HBA": max(
                                    0,
                                    min(
                                        100,
                                        100 - drug_properties["H_Bond_Acceptors"] / 10 * 100
                                    )
                                )
                            }

                            radar_df = pd.DataFrame({
                                "Metric": list(radar_metrics.keys()),
                                "Score": list(radar_metrics.values())
                            })

                            radar_fig = px.line_polar(
                                radar_df,
                                r="Score",
                                theta="Metric",
                                line_close=True,
                                range_r=[0, 100],
                                title="Drug-Likeness Radar Chart"
                            )

                            radar_fig.update_traces(
                                fill="toself"
                            )

                            st.plotly_chart(
                                radar_fig,
                                width="stretch"
                            )

                            st.markdown("---")

                            st.subheader("Molecular Descriptor Table")

                            descriptor_df = pd.DataFrame({
                                "Descriptor": [
                                    "Molecular Weight",
                                    "LogP",
                                    "H-Bond Donors",
                                    "H-Bond Acceptors",
                                    "TPSA",
                                    "Rotatable Bonds",
                                    "Ring Count",
                                    "Heavy Atom Count",
                                    "Synthetic Accessibility Estimate"
                                ],
                                "Value": [
                                    drug_properties["Molecular_Weight"],
                                    drug_properties["LogP"],
                                    drug_properties["H_Bond_Donors"],
                                    drug_properties["H_Bond_Acceptors"],
                                    drug_properties["TPSA"],
                                    drug_properties["Rotatable_Bonds"],
                                    drug_properties["Ring_Count"],
                                    drug_properties["Heavy_Atom_Count"],
                                    drug_properties["Synthetic_Accessibility_Estimate"]
                                ],
                                "Typical Guidance": [
                                    "<= 500 preferred",
                                    "<= 5 preferred",
                                    "<= 5 preferred",
                                    "<= 10 preferred",
                                    "<= 140 often preferred",
                                    "<= 10 often preferred",
                                    "Context dependent",
                                    "Context dependent",
                                    "Lower is easier"
                                ]
                            })

                            descriptor_df["Value"] = descriptor_df[
                                "Value"
                            ].astype(str)

                            st.dataframe(
                                descriptor_df,
                                width="stretch"
                            )

                            st.markdown("---")

                            st.subheader("Rule-Based Drug-Likeness Details")

                            rules_df = pd.DataFrame({
                                "Rule Set": [
                                    "Lipinski",
                                    "Lipinski",
                                    "Lipinski",
                                    "Lipinski",
                                    "Veber",
                                    "Veber",
                                    "Ghose",
                                    "Ghose",
                                    "Ghose",
                                    "Lead-Likeness",
                                    "Lead-Likeness",
                                    "Lead-Likeness"
                                ],
                                "Rule": [
                                    "Molecular Weight <= 500",
                                    "LogP <= 5",
                                    "H-Bond Donors <= 5",
                                    "H-Bond Acceptors <= 10",
                                    "Rotatable Bonds <= 10",
                                    "TPSA <= 140",
                                    "160 <= Molecular Weight <= 480",
                                    "-0.4 <= LogP <= 5.6",
                                    "Heavy Atom Count 20 to 70",
                                    "Molecular Weight <= 350",
                                    "LogP <= 3.5",
                                    "Rotatable Bonds <= 7"
                                ],
                                "Pass": [
                                    drug_properties["Rule_MW_<=500"],
                                    drug_properties["Rule_LogP_<=5"],
                                    drug_properties["Rule_HBD_<=5"],
                                    drug_properties["Rule_HBA_<=10"],
                                    drug_properties["Rule_Veber_RB_<=10"],
                                    drug_properties["Rule_Veber_TPSA_<=140"],
                                    drug_properties["Rule_Ghose_MW_160_480"],
                                    drug_properties["Rule_Ghose_LogP_-0.4_5.6"],
                                    drug_properties["Rule_Ghose_Atom_Count_20_70"],
                                    drug_properties["Rule_Lead_MW_<=350"],
                                    drug_properties["Rule_Lead_LogP_<=3.5"],
                                    drug_properties["Rule_Lead_RB_<=7"]
                                ]
                            })

                            rules_df["Status"] = rules_df["Pass"].apply(
                                lambda x: "Pass" if x else "Fail"
                            )

                            st.dataframe(
                                rules_df,
                                width="stretch"
                            )

                            st.markdown("---")

                            st.subheader("Drug-Likeness Interpretation")

                            interpretation_text = (
                                f"The selected molecule ({drug_name}) has molecular weight "
                                f"{drug_properties['Molecular_Weight']}, LogP "
                                f"{drug_properties['LogP']}, {drug_properties['H_Bond_Donors']} "
                                f"H-bond donor(s), and {drug_properties['H_Bond_Acceptors']} "
                                f"H-bond acceptor(s). It has {drug_properties['Lipinski_Violations']} "
                                f"Lipinski violation(s), {drug_properties['Veber_Violations']} Veber violation(s), "
                                f"{drug_properties['Ghose_Violations']} Ghose violation(s), and "
                                f"{drug_properties['Lead_Likeness_Violations']} lead-likeness violation(s). "
                                f"PAINS status: {drug_properties['PAINS_Status']}. "
                                f"Synthetic accessibility estimate: {drug_properties['Synthetic_Accessibility_Estimate']} "
                                f"({drug_properties['Synthetic_Accessibility_Status']})."
                            )

                            st.info(
                                interpretation_text
                            )

                            st.subheader("Download Reports")

                            report_csv = report_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download Drug-Likeness CSV Report",
                                data=report_csv,
                                file_name="drug_likeness_report.csv",
                                mime="text/csv"
                            )

                            pdf_bytes = create_drug_likeness_pdf_report(
                                molecule_name=drug_name,
                                smiles=drug_smiles,
                                report_df=report_df
                            )

                            st.download_button(
                                label="Download Drug-Likeness PDF Report",
                                data=pdf_bytes,
                                file_name="drug_likeness_report.pdf",
                                mime="application/pdf"
                            )

            # ==================================================
            # BATCH DRUG-LIKENESS ANALYSIS
            # ==================================================

            else:

                st.subheader("Batch Drug-Likeness CSV Analysis")

                st.write(
                    "Upload a CSV containing at least a `SMILES` column. Optional columns: `Molecule_Name` or `Name`."
                )

                uploaded_drug_csv = st.file_uploader(
                    "Upload CSV for Batch Drug-Likeness Analysis",
                    type=["csv"],
                    key="batch_drug_likeness_csv_upload"
                )

                use_dataset_for_batch = st.checkbox(
                    "Use full available dataset instead of uploading CSV",
                    value=False,
                    key="use_full_dataset_for_batch_drug_likeness"
                )

                batch_input_df = None

                if use_dataset_for_batch:

                    batch_input_df = drug_df[
                        [
                            "Molecule_Name",
                            "SMILES"
                        ]
                    ].copy()

                    st.info(
                        f"Using full dataset: {len(batch_input_df)} molecules."
                    )

                elif uploaded_drug_csv is not None:

                    batch_input_df = pd.read_csv(
                        uploaded_drug_csv
                    )

                    st.success(
                        f"Uploaded CSV loaded: {batch_input_df.shape[0]} rows."
                    )

                    st.dataframe(
                        batch_input_df.head(20),
                        width="stretch"
                    )

                if batch_input_df is not None:

                    if "SMILES" not in batch_input_df.columns:

                        st.error(
                            "CSV must contain a `SMILES` column."
                        )

                    else:

                        max_batch_rows = st.slider(
                            "Maximum molecules to analyze",
                            min_value=10,
                            max_value=min(3000, len(batch_input_df)),
                            value=min(500, len(batch_input_df)),
                            step=50,
                            key="batch_drug_likeness_max_rows"
                        )

                        if st.button(
                            "Run Batch Drug-Likeness Analysis",
                            key="run_batch_drug_likeness_analysis"
                        ):

                            with st.spinner(
                                "Running batch drug-likeness analysis..."
                            ):

                                batch_result_df = generate_batch_drug_likeness_dataframe(
                                    batch_input_df.head(max_batch_rows)
                                )

                            if batch_result_df.empty:

                                st.warning(
                                    "No valid molecules were processed."
                                )

                            else:

                                st.success(
                                    f"Batch drug-likeness analysis completed for {len(batch_result_df)} molecules."
                                )

                                st.dataframe(
                                    batch_result_df,
                                    width="stretch"
                                )

                                st.subheader("Batch Summary")

                                col_b1, col_b2, col_b3, col_b4 = st.columns(4)

                                with col_b1:
                                    st.metric(
                                        "Lipinski Pass",
                                        int(
                                            (
                                                batch_result_df["Lipinski_Status"]
                                                ==
                                                "Pass"
                                            ).sum()
                                        )
                                    )

                                with col_b2:
                                    st.metric(
                                        "Veber Pass",
                                        int(
                                            (
                                                batch_result_df["Veber_Status"]
                                                ==
                                                "Pass"
                                            ).sum()
                                        )
                                    )

                                with col_b3:
                                    st.metric(
                                        "No PAINS Alerts",
                                        int(
                                            (
                                                batch_result_df["PAINS_Status"]
                                                ==
                                                "No PAINS Alerts"
                                            ).sum()
                                        )
                                    )

                                with col_b4:
                                    st.metric(
                                        "Lead-like",
                                        int(
                                            (
                                                batch_result_df["Lead_Likeness_Status"]
                                                ==
                                                "Lead-like"
                                            ).sum()
                                        )
                                    )

                                batch_csv = batch_result_df.to_csv(
                                    index=False
                                ).encode("utf-8")

                                st.download_button(
                                    label="Download Batch Drug-Likeness CSV Report",
                                    data=batch_csv,
                                    file_name="batch_drug_likeness_report.csv",
                                    mime="text/csv"
                                )

        except Exception as e:

            st.error(f"Drug-likeness analysis failed: {e}")


    st.markdown("---")

    st.caption(
        "Hybrid GNN AI Cheminformatics Platform | "
        "Enhanced PDF Report + Batch Confidence Report + Dashboard Summary + "
        "Murcko Scaffold Analysis + OOD Detection + Chemical Space PCA + t-SNE + UMAP + Interactive Plotly UMAP + AI Overlay + Drug-Likeness Analysis"
    )


elif st.session_state["authentication_status"] is False:

    st.error("Incorrect username or password")

else:
    st.warning("Please enter username and password")
