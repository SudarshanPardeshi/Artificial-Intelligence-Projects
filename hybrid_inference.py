import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import torch
import pandas as pd
import shap

from torch_geometric.loader import DataLoader

from gnn_utils import mol_to_graph
from gat_model import GATEmbeddingModel
from rdkit_utils import compute_rdkit_descriptors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gat_model = GATEmbeddingModel().to(device)
gat_model.load_state_dict(
    torch.load("gat_model.pth", map_location=device)
)
gat_model.eval()

hybrid_model = joblib.load("hybrid_gat_model.pkl")
hybrid_feature_names = joblib.load("hybrid_feature_names.pkl")


def get_gat_embedding(smiles: str):
    graph = mol_to_graph(smiles)

    loader = DataLoader([graph], batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embedding = gat_model(batch, return_embedding=True)

    return embedding.cpu().numpy()


def prepare_hybrid_features(smiles: str):
    rdkit_df = compute_rdkit_descriptors(smiles)
    gat_embedding = get_gat_embedding(smiles)

    gat_cols = [
        f"GAT_Embedding_{i + 1}"
        for i in range(gat_embedding.shape[1])
    ]

    gat_df = pd.DataFrame(gat_embedding, columns=gat_cols)

    hybrid_df = pd.concat(
        [
            rdkit_df.reset_index(drop=True),
            gat_df.reset_index(drop=True)
        ],
        axis=1
    )

    for col in hybrid_feature_names:
        if col not in hybrid_df.columns:
            hybrid_df[col] = 0

    hybrid_df = hybrid_df[hybrid_feature_names]

    return hybrid_df


def predict_hybrid_gat(smiles: str):
    hybrid_df = prepare_hybrid_features(smiles)
    prediction = hybrid_model.predict(hybrid_df)[0]
    return float(prediction)


def get_hybrid_feature_importance(top_n: int = 15):
    importances = hybrid_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": hybrid_feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(top_n)

    return importance_df


def explain_hybrid_gat_prediction(smiles: str, top_n: int = 10):
    hybrid_df = prepare_hybrid_features(smiles)

    explainer = shap.TreeExplainer(hybrid_model)
    shap_values = explainer.shap_values(hybrid_df)

    explanation_df = pd.DataFrame({
        "Feature": hybrid_df.columns,
        "Feature_Value": hybrid_df.iloc[0].values,
        "SHAP_Value": shap_values[0]
    })

    explanation_df["Abs_SHAP"] = explanation_df["SHAP_Value"].abs()

    explanation_df = explanation_df.sort_values(
        "Abs_SHAP",
        ascending=False
    ).head(top_n)

    return explanation_df[
        ["Feature", "Feature_Value", "SHAP_Value"]
    ]


def explain_hybrid_gat_batch(smiles_list, top_n: int = 5):
    rows = []

    explainer = shap.TreeExplainer(hybrid_model)

    for smiles in smiles_list:
        try:
            hybrid_df = prepare_hybrid_features(smiles)

            prediction_k = float(hybrid_model.predict(hybrid_df)[0])
            prediction_c = prediction_k - 273.15

            shap_values = explainer.shap_values(hybrid_df)

            temp_df = pd.DataFrame({
                "Feature": hybrid_df.columns,
                "Feature_Value": hybrid_df.iloc[0].values,
                "SHAP_Value": shap_values[0]
            })

            temp_df["Abs_SHAP"] = temp_df["SHAP_Value"].abs()

            temp_df = temp_df.sort_values(
                "Abs_SHAP",
                ascending=False
            ).head(top_n)

            row = {
                "SMILES": smiles,
                "Predicted_Melting_Point_K": round(prediction_k, 2),
                "Predicted_Melting_Point_C": round(prediction_c, 2),
                "Status": "Success"
            }

            for i, (_, r) in enumerate(temp_df.iterrows(), start=1):
                row[f"Top_{i}_Feature"] = r["Feature"]
                row[f"Top_{i}_Feature_Value"] = r["Feature_Value"]
                row[f"Top_{i}_SHAP_Value"] = r["SHAP_Value"]

            rows.append(row)

        except Exception as e:
            rows.append({
                "SMILES": smiles,
                "Predicted_Melting_Point_K": None,
                "Predicted_Melting_Point_C": None,
                "Status": f"Failed: {e}"
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    smiles = "CCO"

    prediction_k = predict_hybrid_gat(smiles)
    prediction_c = prediction_k - 273.15

    print("SMILES:", smiles)
    print("Hybrid GAT Prediction:", round(prediction_k, 2), "K")
    print("Hybrid GAT Prediction:", round(prediction_c, 2), "°C")

    print("\nTop Hybrid Feature Importance:\n")
    print(get_hybrid_feature_importance(top_n=10))

    print("\nHybrid GAT SHAP Explanation:\n")
    print(explain_hybrid_gat_prediction(smiles, top_n=10))