import joblib
import pandas as pd
import shap

from rdkit_utils import compute_rdkit_descriptors

model = joblib.load("rdkit_lightgbm_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(model)


def prepare_features(smiles: str):
    features_df = compute_rdkit_descriptors(smiles)

    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0

    return features_df[feature_columns]


def predict_melting_point(smiles: str):
    features_df = prepare_features(smiles)
    return float(model.predict(features_df)[0])


def explain_prediction(smiles: str, top_n: int = 10):
    features_df = prepare_features(smiles)

    shap_values = explainer.shap_values(features_df)

    explanation_df = pd.DataFrame({
        "Feature": features_df.columns,
        "Feature_Value": features_df.iloc[0].values,
        "SHAP_Value": shap_values[0]
    })

    explanation_df["Abs_SHAP"] = explanation_df["SHAP_Value"].abs()

    explanation_df = explanation_df.sort_values(
        "Abs_SHAP",
        ascending=False
    ).head(top_n)

    return explanation_df[["Feature", "Feature_Value", "SHAP_Value"]]


def predict_batch(smiles_list):
    results = []

    for smiles in smiles_list:
        try:
            pred = predict_melting_point(smiles)
            results.append({
                "SMILES": smiles,
                "Predicted_Melting_Point_K": round(pred, 2),
                "Status": "Success"
            })
        except Exception as e:
            results.append({
                "SMILES": smiles,
                "Predicted_Melting_Point_K": None,
                "Status": f"Failed: {e}"
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    test_smiles = ["CCO", "C", "CCN"]

    print(predict_batch(test_smiles))

    print("\nSHAP explanation for CCO:")
    print(explain_prediction("CCO"))