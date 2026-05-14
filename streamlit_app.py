
import os
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

from sklearn.decomposition import PCA
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
    df = pd.read_csv("all_smiles_with_names.csv")
    required_cols = ["Molecule_Name", "SMILES"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df = df[["Molecule_Name", "SMILES"]].copy()
    df = df.dropna(subset=["SMILES"])

    df["Molecule_Name"] = df["Molecule_Name"].fillna("Name Not Found").astype(str)
    df["SMILES"] = df["SMILES"].fillna("").astype(str)
    df["Molecule_Display"] = df["Molecule_Name"] + " | " + df["SMILES"]

    return df


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

    result = AllChem.EmbedMolecule(
        mol,
        AllChem.ETKDG()
    )

    if result != 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass

    return Chem.MolToMolBlock(mol)


def show_3d_molecule(smiles):
    if not PY3DMOL_AVAILABLE:
        st.warning("py3Dmol is not installed. Run: pip install py3Dmol")
        return

    mol_block = create_3d_molblock(smiles)

    if mol_block is None:
        st.warning("3D structure could not be generated for this molecule.")
        return

    viewer = py3Dmol.view(width=700, height=500)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()

    components.html(
        viewer._make_html(),
        height=520,
        width=720
    )


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

    st.write(
        "Predict molecular melting point using RDKit descriptors, LightGBM, Hybrid GAT AI, "
        "Ensemble AI, molecule search, similarity search, PNG export, 3D visualization, "
        "uncertainty estimation, enhanced PDF reporting, batch reports, dashboard analytics, "
        "and Murcko scaffold analysis."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
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
        "Interactive Plotly UMAP + AI Overlay"
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

                selected_index = st.selectbox(
                    "Select IUPAC Name | SMILES",
                    options=smiles_df.index,
                    format_func=lambda x: smiles_df.loc[x, "Molecule_Display"],
                    key="single_selectbox"
                )

                selected_name = smiles_df.loc[selected_index, "Molecule_Name"]
                manual_smiles = smiles_df.loc[selected_index, "SMILES"]

                st.success(f"IUPAC Name: {selected_name}")
                st.success(f"SMILES: {manual_smiles}")

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

        try:
            explorer_df = load_molecule_dataset()

            search_query = st.text_input(
                "Search molecule by IUPAC name or SMILES",
                value="",
                key="molecule_search_input"
            )

            if st.button("Refresh / Reset Search"):
                st.rerun()

            if search_query.strip() != "":
                explorer_df = explorer_df[
                    explorer_df["Molecule_Name"].str.contains(
                        search_query,
                        case=False,
                        na=False,
                        regex=False
                    )
                    |
                    explorer_df["SMILES"].str.contains(
                        search_query,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            if explorer_df.empty:
                st.warning("No molecule found for this search.")
                st.stop()

            st.info(f"Total molecules found: {len(explorer_df)}")

            selected_index = st.selectbox(
                "Select Molecule: IUPAC Name | SMILES",
                options=explorer_df.index,
                format_func=lambda x: explorer_df.loc[x, "Molecule_Display"],
                key="explorer_selectbox"
            )

            explorer_selected_name = explorer_df.loc[selected_index, "Molecule_Name"]
            explorer_selected_smiles = explorer_df.loc[selected_index, "SMILES"]
            safe_name = make_safe_filename(explorer_selected_name)

            st.success(f"IUPAC Name: {explorer_selected_name}")
            st.success(f"SMILES: {explorer_selected_smiles}")

            st.subheader("Copy Selected Molecule Details")

            st.text_input(
                "Copy IUPAC Name",
                value=explorer_selected_name,
                key=f"copy_iupac_name_{selected_index}"
            )

            st.text_input(
                "Copy SMILES",
                value=explorer_selected_smiles,
                key=f"copy_smiles_{selected_index}"
            )

            st.code(
                f"IUPAC Name: {explorer_selected_name}\n"
                f"SMILES: {explorer_selected_smiles}",
                language="text"
            )

            mol = Chem.MolFromSmiles(explorer_selected_smiles)

            if mol is not None:

                st.subheader("2D Molecular Structure")

                molecule_image = Draw.MolToImage(mol, size=(400, 400))
                st.image(molecule_image, caption="2D Molecular Structure")

                img_buffer = BytesIO()
                molecule_image.save(img_buffer, format="PNG")

                st.download_button(
                    label="Download Molecule Image PNG",
                    data=img_buffer.getvalue(),
                    file_name=f"{safe_name}_molecule.png",
                    mime="image/png"
                )

                st.subheader("3D Molecular Visualization")

                show_3d = st.checkbox(
                    "Show Interactive 3D Molecule",
                    value=False
                )

                if show_3d:
                    show_3d_molecule(explorer_selected_smiles)

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

                if st.button("Find Top 10 Similar Molecules"):
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

                        similar_csv = similar_df.to_csv(index=False).encode("utf-8")

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
            success_count = len(dashboard_df[dashboard_df["Prediction Result"] == "Success"])
            failed_count = len(dashboard_df[dashboard_df["Prediction Result"] == "Failed"])
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
                label="Download Dashboard Data CSV",
                data=dashboard_csv,
                file_name="dashboard_summary_data.csv",
                mime="text/csv"
            )

    with tab7:

        st.subheader("Murcko Scaffold Analysis")

        st.write(
            "Analyze molecular core structures using Bemis–Murcko scaffolds. "
            "This helps identify repeated chemical cores, scaffold diversity, "
            "and structure families in the dataset."
        )

        try:
            molecule_df = load_molecule_dataset()

            if st.button("Generate Scaffold Analysis"):

                with st.spinner("Generating Murcko scaffolds..."):
                    scaffold_df = generate_scaffold_dataframe(molecule_df)

                if scaffold_df.empty:
                    st.warning("No valid scaffolds could be generated.")

                else:
                    st.success("Scaffold analysis completed.")

                    total_molecules = len(scaffold_df)
                    unique_scaffolds = scaffold_df["Murcko_Scaffold"].nunique()
                    no_scaffold_count = len(
                        scaffold_df[
                            scaffold_df["Murcko_Scaffold"] == "No Scaffold"
                        ]
                    )

                    col_s1, col_s2, col_s3 = st.columns(3)

                    with col_s1:
                        st.metric("Molecules Analyzed", total_molecules)

                    with col_s2:
                        st.metric("Unique Scaffolds", unique_scaffolds)

                    with col_s3:
                        st.metric("No Scaffold Molecules", no_scaffold_count)

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
                        .sort_values(by="Molecule_Count", ascending=False)
                    )

                    st.dataframe(scaffold_freq_df)

                    st.subheader("Top Scaffold Frequency Plot")

                    top_n_scaffolds = st.slider(
                        "Select number of top scaffolds to display",
                        min_value=5,
                        max_value=30,
                        value=10,
                        step=5
                    )

                    plot_df = scaffold_freq_df.head(top_n_scaffolds).copy()
                    plot_df["Scaffold_Label"] = (
                        plot_df["Murcko_Scaffold"]
                        .astype(str)
                        .str.slice(0, 25)
                    )

                    fig_scaffold, ax_scaffold = plt.subplots(figsize=(10, 5))
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
                            )
                        )

                        scaffold_mol = Chem.MolFromSmiles(selected_scaffold)

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

                            st.dataframe(selected_scaffold_molecules)

                    scaffold_csv = scaffold_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Full Scaffold Analysis CSV",
                        data=scaffold_csv,
                        file_name="murcko_scaffold_analysis.csv",
                        mime="text/csv"
                    )

                    scaffold_freq_csv = scaffold_freq_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Scaffold Frequency CSV",
                        data=scaffold_freq_csv,
                        file_name="murcko_scaffold_frequency.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Scaffold analysis failed: {e}")


    # ==================================================
    # TAB 8 — OOD DETECTION
    # ==================================================

    with tab8:

        st.subheader("Out-of-Distribution (OOD) Detection")

        st.write(
            "This checks whether a molecule is similar to your known dataset chemistry. "
            "It uses Morgan fingerprints and Tanimoto similarity against the full molecule dataset."
        )

        try:

            ood_df = load_molecule_dataset()

            ood_input_mode = st.radio(
                "Choose OOD Input Method",
                [
                    "Select from Dataset",
                    "Enter Custom SMILES"
                ],
                key="ood_input_mode"
            )

            if ood_input_mode == "Select from Dataset":

                selected_ood_index = st.selectbox(
                    "Select Molecule for OOD Detection",
                    options=ood_df.index,
                    format_func=lambda x: ood_df.loc[x, "Molecule_Display"],
                    key="ood_selectbox"
                )

                ood_query_name = ood_df.loc[
                    selected_ood_index,
                    "Molecule_Name"
                ]

                ood_query_smiles = ood_df.loc[
                    selected_ood_index,
                    "SMILES"
                ]

            else:

                ood_query_name = "Custom Input"

                ood_query_smiles = st.text_input(
                    "Enter SMILES for OOD Detection",
                    value="CCO",
                    key="ood_custom_smiles"
                )

            st.success(f"Molecule: {ood_query_name}")
            st.success(f"SMILES: {ood_query_smiles}")

            if st.button("Run OOD Detection", key="run_ood_detection"):

                with st.spinner("Running OOD detection..."):

                    ood_result = detect_ood_molecule(
                        ood_query_smiles,
                        ood_df
                    )

                st.subheader("OOD Detection Result")

                ood_result_df = pd.DataFrame([ood_result])
                st.dataframe(ood_result_df)

                max_similarity = ood_result["Max_Tanimoto_Similarity"]

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
                    st.success(ood_result["Warning"])

                elif ood_result["OOD_Status"] == "Borderline":
                    st.warning(ood_result["Warning"])

                else:
                    st.error(ood_result["Warning"])

                st.subheader("Nearest Dataset Molecule")

                nearest_smiles = ood_result["Nearest_SMILES"]
                nearest_name = ood_result["Nearest_Molecule_Name"]

                if nearest_smiles is not None:

                    nearest_mol = Chem.MolFromSmiles(nearest_smiles)

                    if nearest_mol is not None:

                        nearest_image = Draw.MolToImage(
                            nearest_mol,
                            size=(350, 350)
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

                st.dataframe(guidance_df)

                ood_csv = ood_result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download OOD Detection CSV",
                    data=ood_csv,
                    file_name="ood_detection_result.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"OOD detection failed: {e}")



    # ==================================================
    # TAB 9 — CHEMICAL SPACE PCA VISUALIZATION
    # ==================================================

    with tab9:

        st.subheader("PCA Chemical Space Visualization")

        st.write(
            "Visualize the global molecular distribution of the dataset using "
            "Morgan fingerprints reduced to two PCA components. This helps show "
            "clusters, dataset diversity, and possible outliers."
        )

        try:

            molecule_df = load_molecule_dataset()

            sample_size = st.slider(
                "Number of molecules to visualize",
                min_value=100,
                max_value=min(3000, len(molecule_df)),
                value=min(1000, len(molecule_df)),
                step=100
            )

            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1
            )

            if st.button("Generate PCA Chemical Space Plot"):

                with st.spinner("Generating Morgan fingerprints and PCA projection..."):

                    if sample_size < len(molecule_df):
                        pca_input_df = molecule_df.sample(
                            n=sample_size,
                            random_state=int(random_state)
                        ).copy()
                    else:
                        pca_input_df = molecule_df.copy()

                    pca_df = generate_pca_chemical_space(
                        pca_input_df,
                        n_bits=2048
                    )

                if pca_df.empty:

                    st.warning(
                        "PCA could not be generated. Not enough valid molecules."
                    )

                else:

                    st.success("PCA chemical space generated successfully.")

                    pca1_var = pca_df["PCA1_Explained_Variance"].iloc[0]
                    pca2_var = pca_df["PCA2_Explained_Variance"].iloc[0]

                    col_p1, col_p2, col_p3 = st.columns(3)

                    with col_p1:
                        st.metric(
                            "Molecules Visualized",
                            len(pca_df)
                        )

                    with col_p2:
                        st.metric(
                            "PCA1 Variance",
                            f"{pca1_var}%"
                        )

                    with col_p3:
                        st.metric(
                            "PCA2 Variance",
                            f"{pca2_var}%"
                        )

                    st.subheader("PCA Chemical Space Plot")

                    fig_pca, ax_pca = plt.subplots(figsize=(9, 6))

                    ax_pca.scatter(
                        pca_df["PCA1"],
                        pca_df["PCA2"],
                        alpha=0.65,
                        s=25
                    )

                    ax_pca.set_xlabel(
                        f"PCA1 ({pca1_var}% variance)"
                    )

                    ax_pca.set_ylabel(
                        f"PCA2 ({pca2_var}% variance)"
                    )

                    ax_pca.set_title(
                        "Chemical Space Visualization using PCA"
                    )

                    st.pyplot(fig_pca)

                    st.subheader("PCA Coordinates Table")

                    st.dataframe(
                        pca_df[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "PCA1",
                                "PCA2"
                            ]
                        ]
                    )

                    pca_csv = pca_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download PCA Coordinates CSV",
                        data=pca_csv,
                        file_name="chemical_space_pca_coordinates.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Interpretation: molecules close together in this plot "
                        "have similar Morgan fingerprint patterns. Points far "
                        "from dense regions may represent potential chemical-space outliers."
                    )

        except Exception as e:

            st.error(f"PCA chemical space visualization failed: {e}")



    # ==================================================
    # TAB 10 — CHEMICAL SPACE t-SNE VISUALIZATION
    # ==================================================

    with tab10:

        st.subheader("t-SNE Chemical Space Visualization")

        st.write(
            "Visualize local molecular similarity groups using t-SNE on Morgan fingerprints. "
            "t-SNE is useful for identifying hidden clusters, local structure families, "
            "and possible chemical-space outliers."
        )

        try:

            molecule_df = load_molecule_dataset()

            sample_size_tsne = st.slider(
                "Number of molecules to visualize with t-SNE",
                min_value=100,
                max_value=min(2000, len(molecule_df)),
                value=min(800, len(molecule_df)),
                step=100,
                key="tsne_sample_size"
            )

            perplexity = st.slider(
                "t-SNE perplexity",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                key="tsne_perplexity"
            )

            random_state_tsne = st.number_input(
                "t-SNE random seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="tsne_random_state"
            )

            st.info(
                "Recommended: use 500–1500 molecules for faster t-SNE. "
                "Large sample sizes may take longer."
            )

            if st.button("Generate t-SNE Chemical Space Plot"):

                with st.spinner("Generating Morgan fingerprints and t-SNE projection..."):

                    if sample_size_tsne < len(molecule_df):
                        tsne_input_df = molecule_df.sample(
                            n=sample_size_tsne,
                            random_state=int(random_state_tsne)
                        ).copy()
                    else:
                        tsne_input_df = molecule_df.copy()

                    tsne_df = generate_tsne_chemical_space(
                        tsne_input_df,
                        n_bits=2048,
                        perplexity=perplexity,
                        random_state=int(random_state_tsne)
                    )

                if tsne_df.empty:

                    st.warning(
                        "t-SNE could not be generated. Not enough valid molecules."
                    )

                else:

                    st.success("t-SNE chemical space generated successfully.")

                    col_t1, col_t2 = st.columns(2)

                    with col_t1:
                        st.metric(
                            "Molecules Visualized",
                            len(tsne_df)
                        )

                    with col_t2:
                        st.metric(
                            "Perplexity Used",
                            int(tsne_df["Perplexity"].iloc[0])
                        )

                    st.subheader("t-SNE Chemical Space Plot")

                    fig_tsne, ax_tsne = plt.subplots(figsize=(9, 6))

                    ax_tsne.scatter(
                        tsne_df["TSNE1"],
                        tsne_df["TSNE2"],
                        alpha=0.65,
                        s=25
                    )

                    ax_tsne.set_xlabel("t-SNE 1")
                    ax_tsne.set_ylabel("t-SNE 2")
                    ax_tsne.set_title("Chemical Space Visualization using t-SNE")

                    st.pyplot(fig_tsne)

                    st.subheader("t-SNE Coordinates Table")

                    st.dataframe(
                        tsne_df[
                            [
                                "Molecule_Name",
                                "SMILES",
                                "TSNE1",
                                "TSNE2",
                                "Perplexity"
                            ]
                        ]
                    )

                    tsne_csv = tsne_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download t-SNE Coordinates CSV",
                        data=tsne_csv,
                        file_name="chemical_space_tsne_coordinates.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Interpretation: molecules close together in t-SNE space "
                        "are locally similar by Morgan fingerprint patterns. "
                        "Separated islands may represent different molecular families or scaffolds."
                    )

        except Exception as e:

            st.error(f"t-SNE chemical space visualization failed: {e}")



    # ==================================================
    # TAB 11 — CHEMICAL SPACE UMAP VISUALIZATION
    # ==================================================

    with tab11:

        st.subheader("UMAP Chemical Space Visualization")

        st.write(
            "Visualize molecular chemical space using UMAP on Morgan fingerprints. "
            "UMAP is useful for cluster separation, scaffold-family visualization, "
            "dataset diversity analysis, and chemical-space outlier detection."
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

                molecule_df = load_molecule_dataset()

                sample_size_umap = st.slider(
                    "Number of molecules to visualize with UMAP",
                    min_value=100,
                    max_value=min(3000, len(molecule_df)),
                    value=min(1500, len(molecule_df)),
                    step=100,
                    key="umap_sample_size"
                )

                n_neighbors = st.slider(
                    "UMAP n_neighbors",
                    min_value=5,
                    max_value=100,
                    value=15,
                    step=5,
                    key="umap_n_neighbors"
                )

                min_dist = st.slider(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.10,
                    step=0.01,
                    key="umap_min_dist"
                )

                random_state_umap = st.number_input(
                    "UMAP random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    key="umap_random_state"
                )

                st.info(
                    "Recommended: n_neighbors = 15 and min_dist = 0.1. "
                    "Increase n_neighbors for broader global structure; decrease min_dist for tighter clusters."
                )

                if st.button("Generate UMAP Chemical Space Plot"):

                    with st.spinner("Generating Morgan fingerprints and UMAP projection..."):

                        if sample_size_umap < len(molecule_df):
                            umap_input_df = molecule_df.sample(
                                n=sample_size_umap,
                                random_state=int(random_state_umap)
                            ).copy()
                        else:
                            umap_input_df = molecule_df.copy()

                        umap_df = generate_umap_chemical_space(
                            umap_input_df,
                            n_bits=2048,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            random_state=int(random_state_umap)
                        )

                    if umap_df.empty:

                        st.warning(
                            "UMAP could not be generated. Not enough valid molecules."
                        )

                    else:

                        st.success("UMAP chemical space generated successfully.")

                        col_u1, col_u2, col_u3 = st.columns(3)

                        with col_u1:
                            st.metric(
                                "Molecules Visualized",
                                len(umap_df)
                            )

                        with col_u2:
                            st.metric(
                                "n_neighbors Used",
                                int(umap_df["n_neighbors"].iloc[0])
                            )

                        with col_u3:
                            st.metric(
                                "min_dist Used",
                                f"{float(umap_df['min_dist'].iloc[0]):.2f}"
                            )

                        st.subheader("UMAP Chemical Space Plot")

                        fig_umap, ax_umap = plt.subplots(figsize=(9, 6))

                        ax_umap.scatter(
                            umap_df["UMAP1"],
                            umap_df["UMAP2"],
                            alpha=0.65,
                            s=25
                        )

                        ax_umap.set_xlabel("UMAP 1")
                        ax_umap.set_ylabel("UMAP 2")
                        ax_umap.set_title("Chemical Space Visualization using UMAP")

                        st.pyplot(fig_umap)

                        st.subheader("UMAP Coordinates Table")

                        st.dataframe(
                            umap_df[
                                [
                                    "Molecule_Name",
                                    "SMILES",
                                    "UMAP1",
                                    "UMAP2",
                                    "n_neighbors",
                                    "min_dist"
                                ]
                            ]
                        )

                        umap_csv = umap_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download UMAP Coordinates CSV",
                            data=umap_csv,
                            file_name="chemical_space_umap_coordinates.csv",
                            mime="text/csv"
                        )

                        st.info(
                            "Interpretation: UMAP clusters often represent molecular families "
                            "or scaffold-related groups. Isolated points may indicate possible "
                            "chemical-space outliers."
                        )

            except Exception as e:

                st.error(f"UMAP chemical space visualization failed: {e}")



    # ==================================================
    # TAB 12 — INTERACTIVE PLOTLY UMAP VISUALIZATION
    # ==================================================

    with tab12:

        st.subheader("Interactive Plotly UMAP Visualization")

        st.write(
            "Explore chemical space interactively using UMAP + Plotly. "
            "You can zoom, pan, hover over molecules, inspect SMILES/IUPAC names, "
            "and download the UMAP coordinate table."
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

                molecule_df = load_molecule_dataset()

                sample_size_plotly_umap = st.slider(
                    "Number of molecules for interactive UMAP",
                    min_value=100,
                    max_value=min(3000, len(molecule_df)),
                    value=min(1500, len(molecule_df)),
                    step=100,
                    key="plotly_umap_sample_size"
                )

                plotly_n_neighbors = st.slider(
                    "Interactive UMAP n_neighbors",
                    min_value=5,
                    max_value=100,
                    value=15,
                    step=5,
                    key="plotly_umap_n_neighbors"
                )

                plotly_min_dist = st.slider(
                    "Interactive UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.10,
                    step=0.01,
                    key="plotly_umap_min_dist"
                )

                plotly_random_state = st.number_input(
                    "Interactive UMAP random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    key="plotly_umap_random_state"
                )

                color_option = st.selectbox(
                    "Color points by",
                    [
                        "None",
                        "Murcko Scaffold",
                        "Molecule Name",
                        "Predicted Melting Point",
                        "Confidence %",
                        "OOD Status"
                    ],
                    key="plotly_umap_color_option"
                )

                st.info(
                    "Recommended: use 1000–2000 molecules for smooth interactive performance. "
                    "Hover over any point to view molecule name and SMILES."
                )

                if st.button("Generate Interactive Plotly UMAP + AI Overlay"):

                    with st.spinner("Generating interactive UMAP chemical space..."):

                        if sample_size_plotly_umap < len(molecule_df):
                            plotly_umap_input_df = molecule_df.sample(
                                n=sample_size_plotly_umap,
                                random_state=int(plotly_random_state)
                            ).copy()
                        else:
                            plotly_umap_input_df = molecule_df.copy()

                        plotly_umap_df = generate_umap_chemical_space(
                            plotly_umap_input_df,
                            n_bits=2048,
                            n_neighbors=plotly_n_neighbors,
                            min_dist=plotly_min_dist,
                            random_state=int(plotly_random_state)
                        )

                        if color_option == "Murcko Scaffold":

                            plotly_umap_df["Murcko_Scaffold"] = plotly_umap_df[
                                "SMILES"
                            ].apply(get_murcko_scaffold)

                            scaffold_counts = plotly_umap_df[
                                "Murcko_Scaffold"
                            ].value_counts()

                            top_scaffolds = scaffold_counts.head(10).index.tolist()

                            plotly_umap_df["Scaffold_Group"] = plotly_umap_df[
                                "Murcko_Scaffold"
                            ].apply(
                                lambda x: x if x in top_scaffolds else "Other"
                            )

                            color_column = "Scaffold_Group"

                        elif color_option == "Molecule Name":

                            color_column = "Molecule_Name"

                        elif color_option in [
                            "Predicted Melting Point",
                            "Confidence %",
                            "OOD Status"
                        ]:

                            ai_overlay_rows = []

                            with st.spinner(
                                "Generating AI overlay values. This may take time for larger samples..."
                            ):

                                full_ood_reference_df = load_molecule_dataset()

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
                                            full_ood_reference_df
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

                            if color_option == "Predicted Melting Point":
                                color_column = "Predicted_Melting_Point_K"

                            elif color_option == "Confidence %":
                                color_column = "Confidence_%"

                            else:
                                color_column = "OOD_Status"

                        else:

                            color_column = None

                    if plotly_umap_df.empty:

                        st.warning(
                            "Interactive UMAP could not be generated. Not enough valid molecules."
                        )

                    else:

                        st.success("Interactive Plotly UMAP generated successfully.")

                        col_i1, col_i2, col_i3 = st.columns(3)

                        with col_i1:
                            st.metric(
                                "Molecules Visualized",
                                len(plotly_umap_df)
                            )

                        with col_i2:
                            st.metric(
                                "n_neighbors",
                                int(plotly_umap_df["n_neighbors"].iloc[0])
                            )

                        with col_i3:
                            st.metric(
                                "min_dist",
                                f"{float(plotly_umap_df['min_dist'].iloc[0]):.2f}"
                            )

                        st.subheader("Interactive UMAP Chemical Space")

                        hover_cols = [
                            "Molecule_Name",
                            "SMILES",
                            "n_neighbors",
                            "min_dist"
                        ]

                        optional_hover_cols = [
                            "Murcko_Scaffold",
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

                        fig_interactive = px.scatter(
                            plotly_umap_df,
                            x="UMAP1",
                            y="UMAP2",
                            color=color_column,
                            hover_data=hover_cols,
                            title=f"Interactive UMAP Chemical Space Colored by {color_option}",
                            labels={
                                "UMAP1": "UMAP 1",
                                "UMAP2": "UMAP 2",
                                "Predicted_Melting_Point_K": "Predicted MP (K)",
                                "Confidence_%": "Confidence (%)"
                            },
                            width=1000,
                            height=650,
                            color_continuous_scale="Viridis"
                            if color_option in [
                                "Predicted Melting Point",
                                "Confidence %"
                            ]
                            else None
                        )

                        fig_interactive.update_traces(
                            marker=dict(
                                size=7,
                                opacity=0.75
                            )
                        )

                        fig_interactive.update_layout(
                            legend_title_text=color_option,
                            hovermode="closest"
                        )

                        st.plotly_chart(
                            fig_interactive,
                            use_container_width=True
                        )

                        st.subheader("Interactive UMAP Coordinates Table")

                        display_cols = [
                            "Molecule_Name",
                            "SMILES",
                            "UMAP1",
                            "UMAP2",
                            "n_neighbors",
                            "min_dist"
                        ]

                        optional_display_cols = [
                            "Murcko_Scaffold",
                            "Scaffold_Group",
                            "Predicted_Melting_Point_K",
                            "Predicted_Melting_Point_C",
                            "Confidence_%",
                            "Model_Difference_K",
                            "Estimated_Uncertainty_K",
                            "OOD_Status",
                            "Nearest_Similarity"
                        ]

                        for optional_col in optional_display_cols:
                            if optional_col in plotly_umap_df.columns:
                                display_cols.append(optional_col)

                        st.dataframe(
                            plotly_umap_df[display_cols]
                        )

                        plotly_umap_csv = plotly_umap_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Interactive UMAP Coordinates CSV",
                            data=plotly_umap_csv,
                            file_name="interactive_umap_coordinates.csv",
                            mime="text/csv"
                        )

                        html_buffer = BytesIO()
                        html_content = fig_interactive.to_html(
                            include_plotlyjs="cdn"
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Interactive UMAP Plot HTML",
                            data=html_content,
                            file_name="interactive_umap_plot.html",
                            mime="text/html"
                        )

                        st.info(
                            "Interpretation: clusters may represent related molecular families. "
                            "When colored by prediction, confidence, or OOD status, this plot helps identify "
                            "high/low melting-point regions, unreliable prediction zones, and chemical-space outliers."
                        )

            except Exception as e:

                st.error(f"Interactive Plotly UMAP visualization failed: {e}")


    st.markdown("---")

    st.caption(
        "Hybrid GNN AI Cheminformatics Platform | "
        "Enhanced PDF Report + Batch Confidence Report + Dashboard Summary + "
        "Murcko Scaffold Analysis + OOD Detection + Chemical Space PCA + t-SNE + UMAP + Interactive Plotly UMAP + AI Overlay"
    )


elif st.session_state["authentication_status"] is False:

    st.error("Incorrect username or password")

else:
    st.warning("Please enter username and password")
