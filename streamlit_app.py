
import os
import sqlite3
import hashlib
import secrets
import hmac
import base64
import mimetypes
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
from PIL import Image

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



# ==================================================
# AUTH PHASE 2 — USER HISTORY HELPERS
# ==================================================

def get_current_username():

    return (
        st.session_state.get("username")
        or st.session_state.get("name")
        or "unknown"
    )


def get_current_display_name():

    return (
        st.session_state.get("name")
        or st.session_state.get("username")
        or "User"
    )


def is_admin_user():

    return str(
        st.session_state.get("role", "user")
    ).lower() == "admin"


def filter_logs_for_current_user(rows):

    if rows is None:
        return []

    if is_admin_user():
        return rows

    current_username = str(get_current_username()).lower()

    filtered_rows = []

    for row in rows:

        try:
            row_username = str(row[1]).lower()
        except Exception:
            row_username = ""

        if row_username == current_username:
            filtered_rows.append(row)

    return filtered_rows


def get_user_history_scope_label():

    if is_admin_user():
        return "Admin View: all users"

    return f"My History: {get_current_username()}"



st.set_page_config(
    page_title="Melting Point AI Predictor",
    page_icon="🧪",
    layout="wide"
)


# ==================================================
# AUTHENTICATION PHASE 3 — SQLITE USER REGISTRATION
# ==================================================

AUTH_DB_PATH = "users_auth.db"


def init_auth_db():

    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def hash_password(password, salt=None):

    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return password_hash, salt


def verify_password(password, stored_hash, salt):

    new_hash, _ = hash_password(
        password,
        salt
    )

    return hmac.compare_digest(
        new_hash,
        stored_hash
    )


def get_user_by_username(username):

    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, name, email, password_hash, salt, role
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    row = cursor.fetchone()
    conn.close()

    return row



def load_registered_users():

    conn = sqlite3.connect(AUTH_DB_PATH)

    users_df = pd.read_sql_query(
        """
        SELECT
            id AS User_ID,
            username AS Username,
            name AS Full_Name,
            email AS Email,
            role AS Role,
            created_at AS Created_At
        FROM users
        ORDER BY created_at DESC
        """,
        conn
    )

    conn.close()

    return users_df



def build_admin_monitoring_data():

    users_df = load_registered_users()

    rows = load_prediction_logs()

    if rows is None or len(rows) == 0:

        history_df = pd.DataFrame(
            columns=[
                "ID",
                "Username",
                "SMILES",
                "Model Used",
                "Prediction K",
                "Prediction C",
                "Status",
                "Created At"
            ]
        )

    else:

        history_df = pd.DataFrame(
            rows,
            columns=[
                "ID",
                "Username",
                "SMILES",
                "Model Used",
                "Prediction K",
                "Prediction C",
                "Status",
                "Created At"
            ]
        )

    if not history_df.empty:

        history_df["Prediction K"] = pd.to_numeric(
            history_df["Prediction K"],
            errors="coerce"
        )

        history_df["Prediction C"] = pd.to_numeric(
            history_df["Prediction C"],
            errors="coerce"
        )

        history_df["Created At"] = pd.to_datetime(
            history_df["Created At"],
            errors="coerce"
        )

        history_df["Prediction Result"] = history_df["Status"].apply(
            lambda x: "Success" if str(x).startswith("Success") else "Failed"
        )

    return users_df, history_df


def create_admin_monitoring_report(users_df, history_df):

    summary = {
        "Total Users": len(users_df),
        "Admin Users": int((users_df["Role"] == "admin").sum()) if not users_df.empty else 0,
        "Regular Users": int((users_df["Role"] == "user").sum()) if not users_df.empty else 0,
        "Total Predictions": len(history_df),
        "Successful Predictions": int((history_df["Prediction Result"] == "Success").sum()) if not history_df.empty else 0,
        "Failed Predictions": int((history_df["Prediction Result"] == "Failed").sum()) if not history_df.empty else 0,
    }

    return pd.DataFrame(
        list(summary.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )



def update_user_role(username, new_role):

    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET role = ?
        WHERE username = ?
        """,
        (
            new_role,
            username
        )
    )

    conn.commit()
    conn.close()


def delete_registered_user(username):

    if username == get_current_username():
        return False, "You cannot delete your own active admin account."

    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM users
        WHERE username = ?
        """,
        (
            username,
        )
    )

    conn.commit()
    conn.close()

    return True, f"User '{username}' deleted successfully."




def reset_user_password(username, new_password):

    if new_password is None or new_password.strip() == "":
        return False, "New password cannot be empty."

    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    password_hash, salt = hash_password(new_password)

    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET password_hash = ?,
            salt = ?
        WHERE username = ?
        """,
        (
            password_hash,
            salt,
            username
        )
    )

    affected_rows = cursor.rowcount

    conn.commit()
    conn.close()

    if affected_rows == 0:
        return False, f"User '{username}' not found."

    return True, f"Password reset successfully for user '{username}'."



def create_new_user(username, name, email, password, role="user"):

    username = username.strip()
    name = name.strip()
    email = email.strip().lower()

    if username == "" or name == "" or email == "" or password == "":
        return False, "All fields are required."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    if get_user_by_username(username) is not None:
        return False, "Username already exists."

    password_hash, salt = hash_password(password)

    try:

        conn = sqlite3.connect(AUTH_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users (
                username,
                name,
                email,
                password_hash,
                salt,
                role
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                name,
                email,
                password_hash,
                salt,
                role
            )
        )

        conn.commit()
        conn.close()

        return True, "Account created successfully. Please login."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    except Exception as e:

        return False, f"Registration failed: {e}"


def ensure_default_admin_user():

    # Default admin is created only if database has no admin user.
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM users WHERE role = 'admin'"
    )

    admin_count = cursor.fetchone()[0]
    conn.close()

    if admin_count == 0:

        create_new_user(
            username="admin",
            name="Administrator",
            email="admin@example.com",
            password="admin123",
            role="admin"
        )


def perform_login(username, password):

    user = get_user_by_username(
        username.strip()
    )

    if user is None:
        return False, "Invalid username or password."

    stored_username, name, email, password_hash, salt, role = user

    if verify_password(
        password,
        password_hash,
        salt
    ):

        st.session_state["authentication_status"] = True
        st.session_state["username"] = stored_username
        st.session_state["name"] = name
        st.session_state["email"] = email
        st.session_state["role"] = role

        return True, "Login successful."

    return False, "Invalid username or password."


def logout_user():

    for key in [
        "authentication_status",
        "username",
        "name",
        "email",
        "role"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


init_auth_db()
ensure_default_admin_user()

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

if st.session_state["authentication_status"] is not True:

    st.markdown("## 🔐 Hybrid GNN AI Platform Login")

    login_tab, register_tab = st.tabs(
        [
            "Login",
            "Create Account"
        ]
    )

    with login_tab:

        with st.form("sqlite_login_form"):

            login_username = st.text_input(
                "Username"
            )

            login_password = st.text_input(
                "Password",
                type="password"
            )

            login_submit = st.form_submit_button(
                "Login"
            )

            if login_submit:

                success, message = perform_login(
                    login_username,
                    login_password
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.info(
            "Default admin for first test: username `admin`, password `admin123`. "
            "Please change this later for real deployment."
        )

    with register_tab:

        with st.form("sqlite_registration_form"):

            reg_name = st.text_input(
                "Full Name"
            )

            reg_username = st.text_input(
                "Choose Username"
            )

            reg_email = st.text_input(
                "Email"
            )

            reg_password = st.text_input(
                "Create Password",
                type="password"
            )

            reg_password_confirm = st.text_input(
                "Confirm Password",
                type="password"
            )

            register_submit = st.form_submit_button(
                "Create Account"
            )

            if register_submit:

                if reg_password != reg_password_confirm:

                    st.error(
                        "Passwords do not match."
                    )

                else:

                    success, message = create_new_user(
                        username=reg_username,
                        name=reg_name,
                        email=reg_email,
                        password=reg_password,
                        role="user"
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    st.stop()





# ==================================================
# ENTERPRISE AI STORAGE LAYER — SQLITE
# ==================================================

ENTERPRISE_DB_PATH = "enterprise_ai_storage.db"


def init_enterprise_ai_storage_db():

    conn = sqlite3.connect(ENTERPRISE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            molecule_name TEXT,
            smiles TEXT,
            prediction_type TEXT,
            model_used TEXT,
            predicted_kelvin REAL,
            predicted_celsius REAL,
            confidence REAL,
            confidence_label TEXT,
            uncertainty REAL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_shap_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            smiles TEXT,
            model_name TEXT,
            feature_name TEXT,
            feature_value TEXT,
            shap_value REAL,
            abs_shap REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_uploaded_csvs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            file_name TEXT,
            upload_type TEXT,
            total_rows INTEGER,
            total_columns INTEGER,
            columns_list TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_ood_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            smiles TEXT,
            nearest_molecule_name TEXT,
            nearest_smiles TEXT,
            max_tanimoto_similarity REAL,
            ood_status TEXT,
            reliability TEXT,
            warning TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_report_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            molecule_name TEXT,
            smiles TEXT,
            report_name TEXT,
            report_type TEXT,
            report_size_mb REAL,
            generated_by TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            activity_type TEXT,
            activity_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def save_enterprise_prediction(
    molecule_name,
    smiles,
    prediction_type,
    model_used,
    predicted_kelvin,
    predicted_celsius,
    confidence=None,
    confidence_label=None,
    uncertainty=None,
    status="Success"
):

    try:

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO enterprise_predictions (
                username,
                molecule_name,
                smiles,
                prediction_type,
                model_used,
                predicted_kelvin,
                predicted_celsius,
                confidence,
                confidence_label,
                uncertainty,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                get_current_username(),
                molecule_name,
                smiles,
                prediction_type,
                model_used,
                predicted_kelvin,
                predicted_celsius,
                confidence,
                confidence_label,
                uncertainty,
                status
            )
        )

        conn.commit()
        conn.close()

    except Exception:
        pass


def save_enterprise_shap_outputs(smiles, model_name, shap_df):

    try:

        if shap_df is None or shap_df.empty:
            return

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        for _, row in shap_df.iterrows():

            feature_name = str(
                row.get("Feature", row.get("feature", "Unknown"))
            )

            feature_value = str(
                row.get("Feature_Value", row.get("feature_value", ""))
            )

            shap_value = row.get(
                "SHAP_Value",
                row.get("shap_value", None)
            )

            abs_shap = row.get(
                "Abs_SHAP",
                abs(shap_value) if shap_value is not None else None
            )

            cursor.execute(
                """
                INSERT INTO enterprise_shap_outputs (
                    username,
                    smiles,
                    model_name,
                    feature_name,
                    feature_value,
                    shap_value,
                    abs_shap
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    get_current_username(),
                    smiles,
                    model_name,
                    feature_name,
                    feature_value,
                    shap_value,
                    abs_shap
                )
            )

        conn.commit()
        conn.close()

    except Exception:
        pass


def save_enterprise_uploaded_csv_metadata(file_name, upload_type, dataframe):

    try:

        total_rows = len(dataframe) if dataframe is not None else 0
        total_columns = len(dataframe.columns) if dataframe is not None else 0
        columns_list = ", ".join(dataframe.columns.astype(str).tolist()) if dataframe is not None else ""

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO enterprise_uploaded_csvs (
                username,
                file_name,
                upload_type,
                total_rows,
                total_columns,
                columns_list
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                get_current_username(),
                file_name,
                upload_type,
                total_rows,
                total_columns,
                columns_list
            )
        )

        conn.commit()
        conn.close()

    except Exception:
        pass


def save_enterprise_ood_result(smiles, ood_result):

    try:

        if ood_result is None:
            return

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO enterprise_ood_results (
                username,
                smiles,
                nearest_molecule_name,
                nearest_smiles,
                max_tanimoto_similarity,
                ood_status,
                reliability,
                warning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                get_current_username(),
                smiles,
                ood_result.get("Nearest_Molecule_Name"),
                ood_result.get("Nearest_SMILES"),
                ood_result.get("Max_Tanimoto_Similarity"),
                ood_result.get("OOD_Status"),
                ood_result.get("Reliability"),
                ood_result.get("Warning")
            )
        )

        conn.commit()
        conn.close()

    except Exception:
        pass


def save_enterprise_report_metadata(
    molecule_name,
    smiles,
    report_name,
    report_type,
    report_bytes=None,
    generated_by="Streamlit"
):

    try:

        report_size_mb = (
            round(len(report_bytes) / (1024 * 1024), 4)
            if report_bytes is not None
            else None
        )

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO enterprise_report_metadata (
                username,
                molecule_name,
                smiles,
                report_name,
                report_type,
                report_size_mb,
                generated_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                get_current_username(),
                molecule_name,
                smiles,
                report_name,
                report_type,
                report_size_mb,
                generated_by
            )
        )

        conn.commit()
        conn.close()

    except Exception:
        pass


def save_enterprise_user_activity(activity_type, activity_description):

    try:

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO enterprise_user_activity (
                username,
                activity_type,
                activity_description
            )
            VALUES (?, ?, ?)
            """,
            (
                get_current_username(),
                activity_type,
                activity_description
            )
        )

        conn.commit()
        conn.close()

    except Exception:
        pass


def load_enterprise_table(table_name, limit=1000):

    conn = sqlite3.connect(ENTERPRISE_DB_PATH)

    df = pd.read_sql_query(
        f'SELECT * FROM "{table_name}" ORDER BY id DESC LIMIT {int(limit)}',
        conn
    )

    conn.close()

    return df


def build_enterprise_storage_summary():

    table_names = [
        "enterprise_predictions",
        "enterprise_shap_outputs",
        "enterprise_uploaded_csvs",
        "enterprise_ood_results",
        "enterprise_report_metadata",
        "enterprise_user_activity"
    ]

    rows = []

    conn = sqlite3.connect(ENTERPRISE_DB_PATH)
    cursor = conn.cursor()

    for table_name in table_names:

        try:
            cursor.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            )
            count = cursor.fetchone()[0]
        except Exception:
            count = 0

        rows.append({
            "Table": table_name,
            "Rows": count,
            "Database": ENTERPRISE_DB_PATH
        })

    conn.close()

    return pd.DataFrame(rows)


init_enterprise_ai_storage_db()





# ==================================================
# MODEL MONITORING LAYER — ENTERPRISE HEALTH CHECKS
# ==================================================

def load_model_monitoring_data():

    try:

        init_enterprise_ai_storage_db()

        conn = sqlite3.connect(ENTERPRISE_DB_PATH)

        predictions_df = pd.read_sql_query(
            """
            SELECT *
            FROM enterprise_predictions
            ORDER BY created_at DESC
            """,
            conn
        )

        ood_df = pd.read_sql_query(
            """
            SELECT *
            FROM enterprise_ood_results
            ORDER BY checked_at DESC
            """,
            conn
        )

        activity_df = pd.read_sql_query(
            """
            SELECT *
            FROM enterprise_user_activity
            ORDER BY created_at DESC
            """,
            conn
        )

        conn.close()

    except Exception:

        predictions_df = pd.DataFrame()
        ood_df = pd.DataFrame()
        activity_df = pd.DataFrame()

    if not predictions_df.empty:

        predictions_df["created_at"] = pd.to_datetime(
            predictions_df["created_at"],
            errors="coerce"
        )

        predictions_df["predicted_kelvin"] = pd.to_numeric(
            predictions_df["predicted_kelvin"],
            errors="coerce"
        )

        predictions_df["confidence"] = pd.to_numeric(
            predictions_df["confidence"],
            errors="coerce"
        )

        predictions_df["uncertainty"] = pd.to_numeric(
            predictions_df["uncertainty"],
            errors="coerce"
        )

        predictions_df["date"] = predictions_df["created_at"].dt.date

    if not ood_df.empty:

        ood_df["checked_at"] = pd.to_datetime(
            ood_df["checked_at"],
            errors="coerce"
        )

        ood_df["max_tanimoto_similarity"] = pd.to_numeric(
            ood_df["max_tanimoto_similarity"],
            errors="coerce"
        )

        ood_df["date"] = ood_df["checked_at"].dt.date

    if not activity_df.empty:

        activity_df["created_at"] = pd.to_datetime(
            activity_df["created_at"],
            errors="coerce"
        )

        activity_df["date"] = activity_df["created_at"].dt.date

    return predictions_df, ood_df, activity_df


def calculate_model_health_status(predictions_df, ood_df):

    if predictions_df.empty:

        return {
            "Health Status": "No Data",
            "Risk Level": "Unknown",
            "Recommendation": "Run predictions to activate monitoring."
        }

    total_predictions = len(predictions_df)

    failed_predictions = len(
        predictions_df[
            predictions_df["status"].astype(str).str.contains(
                "Failed",
                case=False,
                na=False
            )
        ]
    )

    failure_rate = (
        failed_predictions / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    avg_confidence = predictions_df["confidence"].dropna().mean()

    low_confidence_count = len(
        predictions_df[
            predictions_df["confidence"].fillna(100) < 70
        ]
    )

    low_confidence_rate = (
        low_confidence_count / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    ood_rate = 0

    if not ood_df.empty and "ood_status" in ood_df.columns:

        ood_count = len(
            ood_df[
                ood_df["ood_status"].astype(str).str.contains(
                    "Out of Distribution|Borderline",
                    case=False,
                    na=False,
                    regex=True
                )
            ]
        )

        ood_rate = (
            ood_count / len(ood_df) * 100
            if len(ood_df) > 0
            else 0
        )

    if failure_rate > 20 or low_confidence_rate > 40 or ood_rate > 50:
        health_status = "Needs Attention"
        risk_level = "High"
        recommendation = "Review failed predictions, low-confidence molecules, and OOD inputs."

    elif failure_rate > 10 or low_confidence_rate > 25 or ood_rate > 30:
        health_status = "Warning"
        risk_level = "Medium"
        recommendation = "Monitor confidence degradation and OOD trends."

    else:
        health_status = "Healthy"
        risk_level = "Low"
        recommendation = "Model behavior appears stable."

    return {
        "Health Status": health_status,
        "Risk Level": risk_level,
        "Recommendation": recommendation
    }


def build_model_monitoring_summary(predictions_df, ood_df):

    if predictions_df.empty:

        return pd.DataFrame({
            "Metric": [
                "Total Predictions",
                "Average Confidence",
                "Average Uncertainty",
                "Failure Rate %",
                "Low Confidence Rate %",
                "OOD / Borderline Rate %"
            ],
            "Value": [
                0,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ]
        })

    total_predictions = len(predictions_df)

    failed_predictions = len(
        predictions_df[
            predictions_df["status"].astype(str).str.contains(
                "Failed",
                case=False,
                na=False
            )
        ]
    )

    low_confidence_count = len(
        predictions_df[
            predictions_df["confidence"].fillna(100) < 70
        ]
    )

    ood_rate = 0

    if not ood_df.empty and "ood_status" in ood_df.columns:

        ood_count = len(
            ood_df[
                ood_df["ood_status"].astype(str).str.contains(
                    "Out of Distribution|Borderline",
                    case=False,
                    na=False,
                    regex=True
                )
            ]
        )

        ood_rate = round(
            ood_count / len(ood_df) * 100,
            2
        ) if len(ood_df) > 0 else 0

    summary = {
        "Total Predictions": total_predictions,
        "Average Confidence": round(
            predictions_df["confidence"].dropna().mean(),
            2
        ) if predictions_df["confidence"].notna().any() else "N/A",
        "Average Uncertainty": round(
            predictions_df["uncertainty"].dropna().mean(),
            2
        ) if predictions_df["uncertainty"].notna().any() else "N/A",
        "Failure Rate %": round(
            failed_predictions / total_predictions * 100,
            2
        ),
        "Low Confidence Rate %": round(
            low_confidence_count / total_predictions * 100,
            2
        ),
        "OOD / Borderline Rate %": ood_rate
    }

    return pd.DataFrame(
        list(summary.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )




# ==================================================
# MOLECULE COMPARISON WORKSPACE HELPERS
# ==================================================

def calculate_basic_rdkit_descriptor_dict(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol)
    }


def calculate_descriptor_similarity_percent(desc_a, desc_b):

    if desc_a is None or desc_b is None:
        return 0.0

    keys = [key for key in desc_a.keys() if key in desc_b]

    if not keys:
        return 0.0

    values_a = np.array([float(desc_a[key]) for key in keys], dtype=float)
    values_b = np.array([float(desc_b[key]) for key in keys], dtype=float)

    combined = np.vstack([values_a, values_b])
    min_vals = combined.min(axis=0)
    max_vals = combined.max(axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

    norm_a = (values_a - min_vals) / ranges
    norm_b = (values_b - min_vals) / ranges

    distance = np.linalg.norm(norm_a - norm_b)
    max_distance = np.sqrt(len(keys))

    similarity = max(0, 1 - (distance / max_distance))

    return round(similarity * 100, 2)


def calculate_structural_similarity_percent(smiles_a, smiles_b):

    fp_a = get_morgan_fingerprint(smiles_a)
    fp_b = get_morgan_fingerprint(smiles_b)

    if fp_a is None or fp_b is None:
        return 0.0

    similarity = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    return round(similarity * 100, 2)


def calculate_scaffold_match_percent(smiles_a, smiles_b):

    scaffold_a = get_murcko_scaffold(smiles_a)
    scaffold_b = get_murcko_scaffold(smiles_b)

    if scaffold_a is None or scaffold_b is None:
        return 0.0, scaffold_a, scaffold_b

    if scaffold_a == scaffold_b:
        return 100.0, scaffold_a, scaffold_b

    if scaffold_a == "No Scaffold" and scaffold_b == "No Scaffold":
        return 100.0, scaffold_a, scaffold_b

    return 0.0, scaffold_a, scaffold_b


def calculate_shap_similarity_percent(shap_a, shap_b):

    try:

        if shap_a is None or shap_b is None or shap_a.empty or shap_b.empty:
            return None

        if "Feature" not in shap_a.columns or "Feature" not in shap_b.columns:
            return None

        if "SHAP_Value" not in shap_a.columns or "SHAP_Value" not in shap_b.columns:
            return None

        merged = pd.merge(
            shap_a[["Feature", "SHAP_Value"]],
            shap_b[["Feature", "SHAP_Value"]],
            on="Feature",
            suffixes=("_A", "_B")
        )

        if len(merged) < 2:
            return None

        corr = merged["SHAP_Value_A"].corr(merged["SHAP_Value_B"])

        if pd.isna(corr):
            return None

        similarity = ((corr + 1) / 2) * 100

        return round(float(similarity), 2)

    except Exception:
        return None


def calculate_chemical_space_distance_score(desc_similarity_percent):

    if desc_similarity_percent >= 80:
        return "Low Distance", 90.0
    elif desc_similarity_percent >= 60:
        return "Moderate Distance", 70.0
    elif desc_similarity_percent >= 40:
        return "High Distance", 45.0
    else:
        return "Very High Distance", 20.0


def calculate_overall_molecule_similarity_score(
    structural_similarity,
    descriptor_similarity,
    scaffold_match,
    chemical_space_score,
    shap_similarity=None
):

    weighted_score = (
        structural_similarity * 0.35
        + descriptor_similarity * 0.25
        + scaffold_match * 0.15
        + chemical_space_score * 0.15
    )

    weight_sum = 0.90

    if shap_similarity is not None:
        weighted_score += shap_similarity * 0.10
        weight_sum = 1.00

    return round(max(0, min(100, weighted_score / weight_sum)), 2)


def build_molecule_comparison_dataframe(desc_a, desc_b):

    rows = []

    if desc_a is None or desc_b is None:
        return pd.DataFrame()

    for key in desc_a.keys():

        rows.append({
            "Descriptor": key,
            "Molecule A": round(float(desc_a[key]), 4),
            "Molecule B": round(float(desc_b.get(key, np.nan)), 4),
            "Absolute Difference": round(
                abs(float(desc_a[key]) - float(desc_b.get(key, np.nan))),
                4
            )
        })

    return pd.DataFrame(rows)




# ==================================================
# EXPLAINABLE AI NARRATIVE HELPERS
# ==================================================

def interpret_descriptor_scientifically(feature_name):

    feature_text = str(feature_name).lower()

    if any(term in feature_text for term in ["molwt", "molecular weight", "weight", "heavyatom"]):
        return (
            "This descriptor relates to molecular size and mass. Larger molecules often "
            "show stronger intermolecular interactions, which may increase melting point."
        )

    if any(term in feature_text for term in ["logp", "hydrophobic", "lipophil"]):
        return (
            "This descriptor relates to hydrophobicity/lipophilicity. Changes in LogP can "
            "affect crystal packing, polarity balance, and intermolecular attraction."
        )

    if any(term in feature_text for term in ["tpsa", "polar", "surface"]):
        return (
            "This descriptor reflects polar surface area. Higher polarity can increase "
            "hydrogen bonding or dipole interactions, often influencing melting point."
        )

    if any(term in feature_text for term in ["hbond", "donor", "acceptor", "hbd", "hba"]):
        return (
            "This descriptor relates to hydrogen-bonding capacity. Strong hydrogen bonding "
            "can stabilize crystal structures and may raise melting point."
        )

    if any(term in feature_text for term in ["ring", "aromatic", "scaffold"]):
        return (
            "This descriptor reflects cyclic or aromatic structural rigidity. Rigid molecules "
            "may pack more efficiently and can show higher melting points."
        )

    if any(term in feature_text for term in ["rotatable", "rotor", "flexib"]):
        return (
            "This descriptor relates to molecular flexibility. Higher flexibility can reduce "
            "packing efficiency and may lower melting point."
        )

    if any(term in feature_text for term in ["fractioncsp3", "csp3"]):
        return (
            "This descriptor reflects 3D saturation/shape. Molecular shape can strongly affect "
            "packing, crystallinity, and thermal behavior."
        )

    return (
        "This descriptor contributes to the model decision and may reflect structural, electronic, "
        "or physicochemical properties relevant to melting point."
    )


def build_xai_driver_summary(explanation_df):

    if explanation_df is None or explanation_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "No local explanation drivers were available."

    driver_df = explanation_df.copy()

    if "SHAP_Value" in driver_df.columns:
        value_col = "SHAP_Value"
    elif "Importance" in driver_df.columns:
        value_col = "Importance"
    else:
        numeric_cols = driver_df.select_dtypes(include=["number"]).columns.tolist()
        value_col = numeric_cols[0] if numeric_cols else None

    if value_col is None:
        return pd.DataFrame(), pd.DataFrame(), "No numeric contribution column was available."

    feature_col = "Feature" if "Feature" in driver_df.columns else driver_df.columns[0]

    driver_df[value_col] = pd.to_numeric(driver_df[value_col], errors="coerce")
    driver_df = driver_df.dropna(subset=[value_col]).copy()

    positive_df = (
        driver_df[driver_df[value_col] > 0]
        .sort_values(by=value_col, ascending=False)
        .head(5)
        .copy()
    )

    negative_df = (
        driver_df[driver_df[value_col] < 0]
        .sort_values(by=value_col, ascending=True)
        .head(5)
        .copy()
    )

    positive_df["Scientific Meaning"] = positive_df[feature_col].apply(
        interpret_descriptor_scientifically
    )

    negative_df["Scientific Meaning"] = negative_df[feature_col].apply(
        interpret_descriptor_scientifically
    )

    if not positive_df.empty:
        top_positive = ", ".join(positive_df[feature_col].astype(str).head(3).tolist())
    else:
        top_positive = "no strong positive drivers"

    if not negative_df.empty:
        top_negative = ", ".join(negative_df[feature_col].astype(str).head(3).tolist())
    else:
        top_negative = "no strong negative drivers"

    summary_text = (
        f"Top positive drivers: {top_positive}. "
        f"Top negative drivers: {top_negative}."
    )

    return positive_df, negative_df, summary_text


def build_local_xai_narrative(
    molecule_name,
    smiles,
    rdkit_prediction,
    hybrid_prediction,
    ensemble_prediction,
    uncertainty_info,
    explanation_df,
    descriptor_snapshot_df
):

    positive_df, negative_df, driver_summary = build_xai_driver_summary(
        explanation_df
    )

    top_feature = "N/A"
    top_value = "N/A"
    top_feature_meaning = "No descriptor interpretation available."

    try:

        if explanation_df is not None and not explanation_df.empty:

            temp_df = explanation_df.copy()

            if "SHAP_Value" in temp_df.columns:
                value_col = "SHAP_Value"
            elif "Importance" in temp_df.columns:
                value_col = "Importance"
            else:
                numeric_cols = temp_df.select_dtypes(include=["number"]).columns.tolist()
                value_col = numeric_cols[0] if numeric_cols else None

            feature_col = "Feature" if "Feature" in temp_df.columns else temp_df.columns[0]

            if value_col is not None:
                temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors="coerce")
                temp_df["ABS_VALUE"] = temp_df[value_col].abs()
                temp_df = temp_df.dropna(subset=["ABS_VALUE"]).sort_values(
                    by="ABS_VALUE",
                    ascending=False
                )

                if not temp_df.empty:
                    top_feature = str(temp_df.iloc[0][feature_col])
                    top_value = round(float(temp_df.iloc[0][value_col]), 4)
                    top_feature_meaning = interpret_descriptor_scientifically(top_feature)

    except Exception:
        pass

    confidence_label = uncertainty_info.get("confidence_label", "Unknown")
    confidence_value = uncertainty_info.get("confidence", "N/A")
    uncertainty_range = uncertainty_info.get("uncertainty_range", "N/A")
    model_difference = uncertainty_info.get("difference", "N/A")

    local_narrative = (
        f"For {molecule_name}, the ensemble model predicts a melting point of "
        f"{ensemble_prediction:.2f} K ({ensemble_prediction - 273.15:.2f} °C). "
        f"The RDKit LightGBM model predicts {rdkit_prediction:.2f} K, while the Hybrid GAT "
        f"model predicts {hybrid_prediction:.2f} K. The model agreement corresponds to "
        f"{confidence_value}% confidence ({confidence_label}) with an estimated uncertainty "
        f"of ±{uncertainty_range} K and a model difference of {model_difference} K."
    )

    driver_narrative = (
        f"The most influential detected feature is '{top_feature}' with contribution value "
        f"{top_value}. {top_feature_meaning} Positive contribution values generally push the "
        f"predicted melting point upward, while negative contribution values push it downward. "
        f"{driver_summary}"
    )

    descriptor_narrative = (
        "From a descriptor interpretation perspective, the model is evaluating molecular size, "
        "polarity, hydrogen-bonding capacity, rigidity, flexibility, and molecular shape. These "
        "properties are chemically meaningful because melting point is strongly influenced by "
        "crystal packing, intermolecular interactions, and structural rigidity."
    )

    scientific_interpretation = (
        f"{local_narrative}\n\n"
        f"{driver_narrative}\n\n"
        f"{descriptor_narrative}\n\n"
        "Scientific caution: this is a model-based explanation, not an experimental proof. "
        "Predictions and explanations should be validated with experimental melting point data "
        "before scientific or industrial use."
    )

    return {
        "Top Feature": top_feature,
        "Top Value": top_value,
        "Top Feature Meaning": top_feature_meaning,
        "Local Narrative": local_narrative,
        "Driver Narrative": driver_narrative,
        "Descriptor Narrative": descriptor_narrative,
        "Scientific Interpretation": scientific_interpretation,
        "Positive Drivers": positive_df,
        "Negative Drivers": negative_df
    }




# ==================================================
# SCIENTIFIC BENCHMARK HELPERS
# ==================================================

def load_scientific_benchmark_results():

    benchmark_candidates = [
        "results/tables/model_benchmark.csv",
        "results/tables/benchmark_results.csv",
        "results/tables/scientific_benchmark.csv",
        "model_benchmark.csv",
        "benchmark_results.csv",
        "scientific_benchmark.csv"
    ]

    for file_path in benchmark_candidates:

        if os.path.exists(file_path):

            try:

                df = pd.read_csv(file_path)

                if not df.empty:
                    return df, file_path

            except Exception:
                pass

    # Auto-fallback: use known project validation scores where available.
    # These values are used only when no benchmark CSV exists.
    fallback_rows = [
        {
            "Model": "RDKit LightGBM",
            "Model_Type": "Classical ML",
            "Validation_Strategy": "Validation Split",
            "MAE": 33.086647,
            "RMSE": np.nan,
            "R2": np.nan,
            "Scientific_Role": "Best available descriptor-based AI model from project validation"
        },
        {
            "Model": "Baseline Ensemble",
            "Model_Type": "Classical Ensemble",
            "Validation_Strategy": "Validation Split",
            "MAE": 52.412123,
            "RMSE": 67.646427,
            "R2": 0.305247,
            "Scientific_Role": "Baseline comparison model from earlier validation"
        },
        {
            "Model": "Hybrid Descriptor + GAT",
            "Model_Type": "Hybrid AI / GNN",
            "Validation_Strategy": "App Inference Available",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Scientific_Role": "Hybrid molecular graph model; add validation scores when available"
        },
        {
            "Model": "Ensemble AI",
            "Model_Type": "Weighted Ensemble",
            "Validation_Strategy": "App Inference Available",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Scientific_Role": "Final deployed ensemble layer; add holdout/scaffold scores when available"
        }
    ]

    fallback_df = pd.DataFrame(fallback_rows)

    try:

        usage_rows = load_prediction_logs()

        total_app_predictions = len(usage_rows) if usage_rows is not None else 0

        fallback_df["App_Prediction_Logs"] = total_app_predictions

    except Exception:

        fallback_df["App_Prediction_Logs"] = 0

    return fallback_df, "Auto fallback: project validation summary + app prediction logs"



def rank_benchmark_models(benchmark_df):

    df = benchmark_df.copy()

    if "MAE" in df.columns:
        df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")

    if "RMSE" in df.columns:
        df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")

    if "R2" in df.columns:
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")

    if "MAE" in df.columns and df["MAE"].notna().any():

        df = df.sort_values(
            by="MAE",
            ascending=True
        ).reset_index(drop=True)

        df["Rank"] = np.arange(
            1,
            len(df) + 1
        )

    else:

        df["Rank"] = np.arange(
            1,
            len(df) + 1
        )

    return df


def build_validation_statistics_table(benchmark_df):

    df = benchmark_df.copy()

    stats_rows = []

    for metric in [
        "MAE",
        "RMSE",
        "R2"
    ]:

        if metric in df.columns:

            values = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                stats_rows.append({
                    "Metric": metric,
                    "Best": round(values.min(), 4) if metric != "R2" else round(values.max(), 4),
                    "Mean": round(values.mean(), 4),
                    "Std": round(values.std(), 4) if len(values) > 1 else 0,
                    "Available_Models": len(values)
                })

            else:

                stats_rows.append({
                    "Metric": metric,
                    "Best": "N/A",
                    "Mean": "N/A",
                    "Std": "N/A",
                    "Available_Models": 0
                })

    return pd.DataFrame(stats_rows)


def build_scaffold_split_template():

    return pd.DataFrame({
        "Validation Item": [
            "Scaffold-aware split",
            "Chemical family separation",
            "OOD-like validation",
            "Generalization check",
            "Recommended metric"
        ],
        "Scientific Meaning": [
            "Train/test molecules grouped by Murcko scaffold",
            "Prevents near-identical chemistry leakage",
            "Tests model performance on unfamiliar scaffolds",
            "More realistic than random split for chemistry",
            "MAE, RMSE, R2, error percentiles"
        ],
        "Status": [
            "Supported / Recommended",
            "Supported / Recommended",
            "Supported / Recommended",
            "Supported / Recommended",
            "MAE preferred for melting point"
        ]
    })


def build_benchmark_interpretation(benchmark_df):

    ranked_df = rank_benchmark_models(
        benchmark_df
    )

    best_model = "N/A"
    best_mae = "N/A"

    if "MAE" in ranked_df.columns and ranked_df["MAE"].notna().any():

        best_row = ranked_df.dropna(
            subset=["MAE"]
        ).iloc[0]

        best_model = str(
            best_row.get("Model", "N/A")
        )

        best_mae = round(
            float(best_row.get("MAE")),
            4
        )

    interpretation = (
        f"The benchmark section compares classical descriptor-based machine learning, "
        f"hybrid graph-aware AI, and ensemble prediction strategies. "
        f"The current best model by MAE is {best_model} with MAE = {best_mae}. "
        f"Lower MAE/RMSE indicates better prediction accuracy, while higher R² indicates "
        f"better explained variance. Scaffold-aware validation is recommended because "
        f"chemical datasets can suffer from structural leakage under random splitting."
    )

    return interpretation


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def apply_visible_ui_upgrade():

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f5ff 0%, #ffffff 45%, #eef2ff 100%);
        }

        .block-container {
            padding-top: 2.25rem !important;
            max-width: 1540px;
        }

        .safe-logo-card {
            background: #ffffff;
            border: 1.5px solid #c4b5fd;
            border-radius: 22px;
            width: 168px;
            height: 174px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 14px 34px rgba(79,70,229,0.16);
            margin-top: 18px;
        }

        .safe-logo-card img {
            width: 128px;
            height: 128px;
            object-fit: contain;
        }

        .safe-logo-caption {
            text-align: center;
            color: #312e81;
            font-weight: 900;
            font-size: 0.78rem;
            margin-top: 8px;
            width: 168px;
        }

        .safe-hero-card {
            background: linear-gradient(125deg, #2e1065 0%, #4338ca 48%, #2563eb 100%);
            color: #ffffff;
            border-radius: 24px;
            padding: 36px 38px 34px 38px;
            min-height: 228px;
            box-shadow: 0 18px 42px rgba(37,99,235,0.22);
            border: 1px solid rgba(255,255,255,0.12);
            margin-top: 18px;
            overflow: visible;
        }

        .safe-hero-title {
            color: #ffffff !important;
            font-size: clamp(2rem, 3vw, 3.05rem);
            line-height: 1.08;
            font-weight: 950;
            letter-spacing: -0.055em;
            margin: 0 0 18px 0;
        }

        .safe-hero-subtitle {
            font-size: 1.06rem;
            line-height: 1.58;
            color: rgba(255,255,255,0.96);
            margin-bottom: 26px;
            max-width: 1040px;
        }

        .safe-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #ffffff;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.24);
            border-radius: 14px;
            padding: 11px 17px;
            margin: 5px 9px 5px 0;
            font-size: 0.86rem;
            font-weight: 850;
            white-space: nowrap;
        }

        .safe-platform-overview {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 24px;
            padding: 26px 30px;
            margin: 22px 0 22px 0;
            box-shadow: 0 16px 38px rgba(15,23,42,0.07);
        }

        .safe-section-title {
            color: #172033;
            font-size: 1.55rem;
            font-weight: 950;
            margin-bottom: 8px;
            border-left: 5px solid #4f46e5;
            padding-left: 14px;
        }

        .safe-section-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-bottom: 22px;
        }

        .module-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 22px 20px;
            min-height: 126px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            margin-bottom: 12px;
        }

        .module-card-title {
            color: #172033;
            font-weight: 900;
            font-size: 0.92rem;
            margin-top: 6px;
        }

        .module-card-value {
            color: #4f46e5;
            font-weight: 950;
            font-size: 1.82rem;
            line-height: 1.05;
            letter-spacing: -0.035em;
        }

        .module-card-caption {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 9px;
            line-height: 1.35;
        }

        .safe-feature-strip {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 16px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.045);
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
        }

        .safe-feature-item {
            padding: 6px 14px;
            border-right: 1px solid #e5e7eb;
        }

        .safe-feature-item:last-child {
            border-right: none;
        }

        .safe-feature-title {
            color: #312e81;
            font-weight: 950;
            margin-bottom: 5px;
        }

        .safe-feature-subtitle {
            color: #475569;
            font-size: 0.82rem;
        }

        div[data-testid="stTabs"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 8px 10px 0 10px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.055);
            margin-bottom: 20px;
        }

        button[data-baseweb="tab"] {
            font-weight: 850 !important;
            border-radius: 12px 12px 0 0 !important;
            border: 0 !important;
            padding: 11px 15px !important;
            color: #172033 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #4f46e5 !important;
            border-bottom: 4px solid #4f46e5 !important;
            background: rgba(79,70,229,0.04) !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 16px;
            border-radius: 18px;
            box-shadow: 0 8px 20px rgba(15,23,42,0.05);
        }

        .footer-upgraded {
            background: linear-gradient(135deg, #2e1065 0%, #312e81 100%);
            color: white;
            padding: 16px 20px;
            border-radius: 16px;
            text-align: center;
            margin-top: 28px;
            font-size: 0.88rem;
        }

        @media (max-width: 900px) {
            .safe-feature-strip {
                grid-template-columns: 1fr;
            }
            .safe-feature-item {
                border-right: none;
                border-bottom: 1px solid #e5e7eb;
            }
        }

        /* Extra top-safety spacing so Streamlit never clips hero/logo */
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0.6rem !important;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        </style>
        """,
        unsafe_allow_html=True
    )



def load_platform_logo_image():

    logo_paths = [
        "assets/Project Logo.jpg",
        "assets/project_logo.jpg",
        "Project Logo.jpg",
        "project_logo.jpg"
    ]

    for logo_path in logo_paths:

        if os.path.exists(logo_path):

            try:
                return Image.open(logo_path)
            except Exception:
                return logo_path

    return None



def get_platform_logo_data_uri():

    logo_paths = [
        "assets/Project Logo.jpg",
        "assets/project_logo.jpg",
        "Project Logo.jpg",
        "project_logo.jpg"
    ]

    for logo_path in logo_paths:

        if os.path.exists(logo_path):

            try:

                with open(logo_path, "rb") as image_file:
                    encoded_logo = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                mime_type = mimetypes.guess_type(
                    logo_path
                )[0] or "image/jpeg"

                return f"data:{mime_type};base64,{encoded_logo}"

            except Exception:
                return None

    return None



def render_visible_platform_header():

    try:
        _df = load_molecule_dataset()
        total_rows = len(_df)
        unique_molecules = _df["SMILES"].astype(str).nunique() if "SMILES" in _df.columns else total_rows
        valid_smiles = _df["SMILES"].dropna().astype(str).nunique() if "SMILES" in _df.columns else total_rows

        try:
            unique_scaffolds = _df["SMILES"].dropna().astype(str).apply(get_murcko_scaffold).nunique()
        except Exception:
            unique_scaffolds = "N/A"

    except Exception:
        total_rows = "N/A"
        unique_molecules = "N/A"
        valid_smiles = "N/A"
        unique_scaffolds = "N/A"

    logo_data_uri = get_platform_logo_data_uri()

    logo_col, hero_col = st.columns([0.16, 0.84])

    with logo_col:

        if logo_data_uri is not None:

            st.markdown(
                f"""
                <div class="safe-logo-card">
                    <img src="{logo_data_uri}">
                </div>
                <div class="safe-logo-caption">MSG Research Identity</div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="safe-logo-card">
                    <div style="
                        background: linear-gradient(135deg,#2e1065,#2563eb);
                        color:white;
                        border-radius:22px;
                        width:118px;
                        height:118px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-weight:950;
                        font-size:2.2rem;
                    ">MSG</div>
                </div>
                <div class="safe-logo-caption">MSG Research Identity</div>
                """,
                unsafe_allow_html=True
            )

    with hero_col:

        st.markdown(
            f"""
            <div class="safe-hero-card">
                <div class="safe-hero-title">Hybrid GNN AI Cheminformatics Platform</div>
                <div class="safe-hero-subtitle">
                    Molecular melting point prediction, 3D visualization, OOD reliability,
                    chemical-space analytics, scaffold exploration, explainable AI, and drug-likeness screening.
                </div>
                <div>
                    <span class="safe-badge">📦 Dataset: {unique_molecules} molecules</span>
                    <span class="safe-badge">🧠 RDKit + GNN + Ensemble</span>
                    <span class="safe-badge">🧬 PCA · t-SNE · UMAP</span>
                    <span class="safe-badge">🔬 XAI + Drug Discovery</span>
                    <span class="safe-badge">🚀 Deployment Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="safe-platform-overview">
            <div class="safe-section-title">Platform Overview</div>
            <div class="safe-section-subtitle">
                An end-to-end AI platform for molecular property prediction, visualization,
                interpretability, database storage, monitoring, and chemical-space exploration.
            </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="module-card-value">{unique_molecules}</div>
                <div class="module-card-title">Unique Molecules</div>
                <div class="module-card-caption">Distinct molecules in dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="module-card-value" style="color:#16a34a;">{total_rows}</div>
                <div class="module-card-title">Total Dataset Rows</div>
                <div class="module-card-caption">Total records / rows in dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="module-card-value" style="color:#2563eb;">{valid_smiles}</div>
                <div class="module-card-title">Valid SMILES</div>
                <div class="module-card-caption">Valid molecular structures</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="module-card">
                <div class="module-card-value" style="color:#f97316;">{unique_scaffolds}</div>
                <div class="module-card-title">Unique Scaffolds</div>
                <div class="module-card-caption">Unique Bemis-Murcko scaffolds</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
            <div class="safe-feature-strip">
                <div class="safe-feature-item">
                    <div class="safe-feature-title">⚙️ AI Models</div>
                    <div class="safe-feature-subtitle">RDKit, GNN, Ensemble</div>
                </div>
                <div class="safe-feature-item">
                    <div class="safe-feature-title">📈 Visualization</div>
                    <div class="safe-feature-subtitle">2D, 3D, PCA, t-SNE, UMAP</div>
                </div>
                <div class="safe-feature-item">
                    <div class="safe-feature-title">🧬 Explainability</div>
                    <div class="safe-feature-subtitle">SHAP, Local & Global XAI</div>
                </div>
                <div class="safe-feature-item">
                    <div class="safe-feature-title">🛡️ Reliability</div>
                    <div class="safe-feature-subtitle">OOD Detection, Uncertainty</div>
                </div>
                <div class="safe-feature-item">
                    <div class="safe-feature-title">💊 Drug Discovery</div>
                    <div class="safe-feature-subtitle">ADMET, Lipinski, Bioactivity</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



def render_visible_footer():

    st.markdown(
        """
        <div class="footer-upgraded">
            Hybrid GNN AI Cheminformatics Platform · Streamlit · RDKit · PyTorch Geometric · Plotly · UMAP · LightGBM
        </div>
        """,
        unsafe_allow_html=True
    )




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



def resolve_molecule_name_to_smiles(
    molecule_name,
    molecule_df=None
):
    molecule_name_clean = str(molecule_name).strip()

    if molecule_name_clean == "":
        return None, None, "Empty molecule name"

    # 1. Search local full molecule catalog first
    try:
        if molecule_df is None:
            molecule_df = load_molecule_dataset()

        exact_match = molecule_df[
            molecule_df["Molecule_Name"].astype(str).str.lower()
            ==
            molecule_name_clean.lower()
        ]

        if not exact_match.empty:
            return (
                exact_match.iloc[0]["SMILES"],
                exact_match.iloc[0]["Molecule_Name"],
                "Local catalog exact match"
            )

        partial_match = molecule_df[
            molecule_df["Molecule_Name"].astype(str).str.contains(
                molecule_name_clean,
                case=False,
                na=False,
                regex=False
            )
        ]

        if not partial_match.empty:
            return (
                partial_match.iloc[0]["SMILES"],
                partial_match.iloc[0]["Molecule_Name"],
                "Local catalog partial match"
            )

    except Exception:
        pass

    # 2. PubChem fallback
    try:
        pubchem_smiles = name_to_smiles(
            molecule_name_clean
        )

        if pubchem_smiles is not None and str(pubchem_smiles).strip() != "":
            return (
                pubchem_smiles,
                molecule_name_clean,
                "PubChem match"
            )

    except Exception:
        pass

    return None, molecule_name_clean, "Not found"



def display_paginated_dataframe(
    df,
    table_key,
    rows_per_page=100
):
    if df is None or df.empty:
        st.warning("No data available to display.")
        return df

    total_rows = len(df)

    st.markdown("#### Dataset Pagination")

    col_page_size, col_page_number = st.columns([1, 1])

    with col_page_size:

        selected_rows_per_page = st.selectbox(
            "Rows per page",
            options=[25, 50, 100, 200, 500],
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

    start_idx = (page_number - 1) * selected_rows_per_page
    end_idx = start_idx + selected_rows_per_page

    paged_df = df.iloc[start_idx:end_idx].copy()

    st.dataframe(
        paged_df,
        width="stretch"
    )

    st.caption(
        f"Showing rows {start_idx + 1} to "
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
        return None, "Invalid SMILES"

    mol_h = Chem.AddHs(mol)

    # Attempt 1: true 3D conformer using ETKDGv3
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.maxAttempts = 2000

        result = AllChem.EmbedMolecule(
            mol_h,
            params
        )

        if result == 0:

            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    AllChem.MMFFOptimizeMolecule(
                        mol_h,
                        maxIters=1000
                    )
                else:
                    AllChem.UFFOptimizeMolecule(
                        mol_h,
                        maxIters=1000
                    )
            except Exception:
                pass

            return Chem.MolToMolBlock(mol_h), "True 3D conformer"

    except Exception:
        pass

    # Attempt 2: multiple conformers
    try:
        conf_ids = AllChem.EmbedMultipleConfs(
            mol_h,
            numConfs=10,
            randomSeed=42,
            useRandomCoords=True,
            maxAttempts=2000
        )

        if len(conf_ids) > 0:

            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    AllChem.MMFFOptimizeMoleculeConfs(
                        mol_h,
                        maxIters=1000
                    )
                else:
                    AllChem.UFFOptimizeMoleculeConfs(
                        mol_h,
                        maxIters=1000
                    )
            except Exception:
                pass

            return (
                Chem.MolToMolBlock(
                    mol_h,
                    confId=int(conf_ids[0])
                ),
                "True 3D conformer"
            )

    except Exception:
        pass

    # Attempt 3: interactive 2D-coordinate fallback
    try:
        mol_2d = Chem.MolFromSmiles(smiles)

        if mol_2d is None:
            return None, "Invalid SMILES"

        AllChem.Compute2DCoords(mol_2d)

        return Chem.MolToMolBlock(mol_2d), "2D fallback viewer"

    except Exception:
        return None, "3D generation failed"


def show_3d_molecule(smiles, width=650, height=480, viewer_key=None):

    if not PY3DMOL_AVAILABLE:
        st.warning(
            "py3Dmol is not installed. Please run: pip install py3Dmol"
        )
        return

    molblock_result = create_3d_molblock(smiles)

    if isinstance(molblock_result, tuple):
        mol_block, generation_mode = molblock_result
    else:
        mol_block = molblock_result
        generation_mode = "Unknown"

    if mol_block is None:
        st.warning(
            "Molecular viewer could not be generated for this molecule."
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

        if generation_mode == "True 3D conformer":
            st.success("True 3D conformer generated successfully.")
        elif generation_mode == "2D fallback viewer":
            st.warning(
                "True 3D conformer could not be embedded, so an interactive fallback viewer is shown."
            )
        else:
            st.info(f"Viewer mode: {generation_mode}")

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

        report_df_safe = pd.DataFrame(
            table_data[1:],
            columns=table_data[0]
        )

        report_table = safe_pdf_table(
            dataframe=report_df_safe,
            styles=styles,
            max_columns=4,
            page_width=500
        )

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


def get_pdf_logo_path():

    logo_paths = [
        "assets/Project Logo.jpg",
        "assets/project_logo.jpg",
        "Project Logo.jpg",
        "project_logo.jpg"
    ]

    for logo_path in logo_paths:

        if os.path.exists(logo_path):
            return logo_path

    return None


def pdf_footer(canvas, doc):

    canvas.saveState()

    page_width, page_height = A4
    logo_path = get_pdf_logo_path()

    # Soft central watermark on every page
    try:

        if logo_path is not None:

            canvas.saveState()

            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(0.07)

            canvas.drawImage(
                logo_path,
                x=(page_width - 280) / 2,
                y=(page_height - 280) / 2,
                width=280,
                height=280,
                preserveAspectRatio=True,
                mask="auto"
            )

            canvas.restoreState()

            canvas.saveState()

            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(0.85)

            canvas.drawImage(
                logo_path,
                x=page_width - 86,
                y=40,
                width=42,
                height=42,
                preserveAspectRatio=True,
                mask="auto"
            )

            canvas.restoreState()

    except Exception:
        pass

    # Premium footer
    canvas.setStrokeColor(colors.HexColor("#4F46E5"))
    canvas.setLineWidth(1)
    canvas.line(40, 35, page_width - 40, 35)

    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#312E81"))
    canvas.drawString(
        40,
        22,
        "Hybrid GNN AI Cheminformatics Platform | MSG Research Identity"
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(
        page_width - 40,
        22,
        f"Page {doc.page}"
    )

    canvas.restoreState()



# ==================================================
# GLOBAL SAFE PDF TABLE ENGINE
# ==================================================

def _safe_pdf_cell_text(value, max_chars=58):

    if pd.isna(value):
        return ""

    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"

    if isinstance(value, (int, np.integer)):
        return str(value)

    value_text = str(value)

    if len(value_text) > max_chars:
        value_text = value_text[:max_chars - 3] + "..."

    return value_text


def safe_pdf_table(
    dataframe,
    styles,
    theme_header=None,
    theme_light=None,
    theme_border=None,
    max_columns=6,
    max_rows=None,
    page_width=500,
    font_size=7,
    header_font_size=7
):
    """
    Global safe PDF table engine.
    Prevents report cutting by wrapping text, fitting page width,
    repeating headers, and limiting very wide tables.
    """

    if dataframe is None or dataframe.empty:
        dataframe = pd.DataFrame({"Message": ["No data available"]})
    else:
        dataframe = dataframe.copy()

    if max_rows is not None:
        dataframe = dataframe.head(max_rows).copy()

    if len(dataframe.columns) > max_columns:
        dataframe = dataframe.iloc[:, :max_columns].copy()
        dataframe["Note"] = "Extra columns omitted for PDF width safety"

    if theme_header is None:
        theme_header = colors.HexColor("#312E81")

    if theme_light is None:
        theme_light = colors.HexColor("#EEF2FF")

    if theme_border is None:
        theme_border = colors.HexColor("#4F46E5")

    cell_style = ParagraphStyle(
        "SafePDFCellStyle",
        parent=styles["BodyText"],
        fontSize=font_size,
        leading=font_size + 2,
        wordWrap="CJK",
        textColor=colors.HexColor("#0F172A")
    )

    header_style = ParagraphStyle(
        "SafePDFHeaderStyle",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=header_font_size,
        leading=header_font_size + 2,
        wordWrap="CJK",
        textColor=colors.white
    )

    table_data = [
        [
            Paragraph(
                _safe_pdf_cell_text(col, max_chars=35),
                header_style
            )
            for col in dataframe.columns
        ]
    ]

    for _, row in dataframe.iterrows():

        table_data.append([
            Paragraph(
                _safe_pdf_cell_text(row[col], max_chars=58),
                cell_style
            )
            for col in dataframe.columns
        ])

    ncols = max(1, len(dataframe.columns))

    if ncols == 1:
        col_widths = [page_width]
    elif ncols == 2:
        col_widths = [page_width * 0.42, page_width * 0.58]
    elif ncols == 3:
        col_widths = [page_width * 0.34, page_width * 0.33, page_width * 0.33]
    elif ncols == 4:
        col_widths = [page_width * 0.30, page_width * 0.23, page_width * 0.23, page_width * 0.24]
    else:
        col_widths = [page_width / ncols] * ncols

    table = Table(
        table_data,
        colWidths=col_widths,
        repeatRows=1,
        splitByRow=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), theme_header),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, theme_light]),
        ("GRID", (0, 0), (-1, -1), 0.30, colors.HexColor("#CBD5E1")),
        ("BOX", (0, 0), (-1, -1), 0.70, theme_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
    ]))

    return table


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
        rightMargin=38,
        leftMargin=38,
        topMargin=34,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    theme_indigo = colors.HexColor("#312E81")
    theme_blue = colors.HexColor("#2563EB")
    theme_purple = colors.HexColor("#4F46E5")
    theme_light = colors.HexColor("#EEF2FF")
    theme_soft = colors.HexColor("#EDE9FE")
    theme_text = colors.HexColor("#0F172A")
    theme_muted = colors.HexColor("#64748B")
    theme_border = colors.HexColor("#C4B5FD")

    title_style = ParagraphStyle(
        "PremiumPDFTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=theme_indigo,
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "PremiumPDFSubtitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=theme_blue,
        alignment=1,
        spaceAfter=4
    )

    section_style = ParagraphStyle(
        "PremiumSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=theme_indigo,
        spaceBefore=8,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "PremiumBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=theme_text
    )

    small_style = ParagraphStyle(
        "PremiumSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=theme_muted
    )

    white_small_style = ParagraphStyle(
        "WhiteSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    story = []
    logo_path = get_pdf_logo_path()

    # --------------------------------------------------
    # Premium branded header card
    # --------------------------------------------------

    logo_cell = []

    if logo_path is not None:

        try:
            logo_cell.append(
                Image(
                    logo_path,
                    width=62,
                    height=62
                )
            )
        except Exception:
            logo_cell.append(
                Paragraph("MSG", title_style)
            )

    else:

        logo_cell.append(
            Paragraph("MSG", title_style)
        )

    header_text = [
        Paragraph(
            "AI-Based Melting Point Prediction Report",
            title_style
        ),
        Paragraph(
            "Hybrid GNN AI Cheminformatics Platform",
            subtitle_style
        ),
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            small_style
        )
    ]

    header_table = Table(
        [[logo_cell, header_text]],
        colWidths=[82, 420]
    )

    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), theme_light),
        ("BOX", (0, 0), (-1, -1), 1, theme_border),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, theme_purple),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
    ]))

    story.append(header_table)
    story.append(Spacer(1, 12))

    # --------------------------------------------------
    # KPI summary band
    # --------------------------------------------------

    kpi_data = [
        [
            Paragraph("<b>Model</b>", white_small_style),
            Paragraph("<b>Prediction K</b>", white_small_style),
            Paragraph("<b>Prediction C</b>", white_small_style),
            Paragraph("<b>Confidence</b>", white_small_style)
        ],
        [
            Paragraph(str(model_used), body_style),
            Paragraph(f"{prediction_k:.2f} K", body_style),
            Paragraph(f"{prediction_c:.2f} C", body_style),
            Paragraph(f"{confidence}% - {confidence_label}", body_style)
        ]
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[130, 115, 115, 142]
    )

    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), theme_indigo),
        ("BACKGROUND", (0, 1), (-1, 1), theme_soft),
        ("BOX", (0, 0), (-1, -1), 0.8, theme_purple),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # --------------------------------------------------
    # Prediction summary table
    # --------------------------------------------------

    molecule_df = pd.DataFrame({
        "Field": [
            "Molecule Name / IUPAC",
            "SMILES",
            "Model Used",
            "Predicted Melting Point (K)",
            "Predicted Melting Point (C)",
            "Confidence",
            "Confidence Category",
            "Model Difference",
            "Estimated Uncertainty"
        ],
        "Value": [
            molecule_name,
            smiles,
            model_used,
            f"{prediction_k:.2f} K",
            f"{prediction_c:.2f} C",
            f"{confidence}%",
            confidence_label,
            f"{model_difference:.2f} K",
            f"+/- {uncertainty_range:.2f} K"
        ]
    })

    story.append(Paragraph("Prediction Summary", section_style))

    molecule_table = safe_pdf_table(
        dataframe=molecule_df,
        styles=styles,
        theme_header=theme_indigo,
        theme_light=theme_light,
        theme_border=theme_purple,
        max_columns=2,
        page_width=502,
        font_size=8,
        header_font_size=8
    )

    story.append(molecule_table)
    story.append(Spacer(1, 14))

    # --------------------------------------------------
    # Molecular structure card
    # --------------------------------------------------

    if molecule_image is not None:

        img_buffer = BytesIO()
        molecule_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        story.append(Paragraph("Molecular Structure", section_style))

        structure_table = Table(
            [[Image(img_buffer, width=220, height=220)]],
            colWidths=[502]
        )

        structure_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, theme_border),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12)
        ]))

        story.append(structure_table)
        story.append(Spacer(1, 14))

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------

    if model_comparison_df is not None and not model_comparison_df.empty:

        story.append(Paragraph("Model Comparison Table", section_style))

        model_table = safe_pdf_table(
            dataframe=model_comparison_df,
            styles=styles,
            theme_header=theme_indigo,
            theme_light=theme_light,
            theme_border=theme_purple,
            max_columns=4,
            page_width=502
        )

        story.append(model_table)
        story.append(Spacer(1, 14))

    # --------------------------------------------------
    # RDKit properties
    # --------------------------------------------------

    story.append(Paragraph("RDKit Molecular Properties", section_style))

    prop_table = safe_pdf_table(
        dataframe=properties_df,
        styles=styles,
        theme_header=theme_indigo,
        theme_light=theme_light,
        theme_border=theme_purple,
        max_columns=4,
        page_width=502
    )

    story.append(prop_table)
    story.append(Spacer(1, 14))

    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------

    if shap_df is not None and not shap_df.empty:

        story.append(Paragraph("Top SHAP Features", section_style))

        shap_display_df = clean_shap_dataframe(
            shap_df,
            top_n=10
        )

        preferred_shap_cols = [
            col for col in [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
                "Abs_SHAP"
            ]
            if col in shap_display_df.columns
        ]

        if preferred_shap_cols:
            shap_display_df = shap_display_df[
                preferred_shap_cols
            ].copy()

        shap_table = safe_pdf_table(
            dataframe=shap_display_df,
            styles=styles,
            theme_header=theme_indigo,
            theme_light=theme_light,
            theme_border=theme_purple,
            max_columns=4,
            max_rows=12,
            page_width=502,
            font_size=7,
            header_font_size=7
        )

        story.append(shap_table)
        story.append(Spacer(1, 14))

    # --------------------------------------------------
    # Similar molecules
    # --------------------------------------------------

    if similar_df is not None and not similar_df.empty:

        story.append(Paragraph("Top 10 Similar Molecules", section_style))

        similar_table = safe_pdf_table(
            dataframe=similar_df,
            styles=styles,
            theme_header=theme_indigo,
            theme_light=theme_light,
            theme_border=theme_purple,
            max_columns=5,
            max_rows=10,
            page_width=502,
            font_size=6,
            header_font_size=6
        )

        story.append(similar_table)
        story.append(Spacer(1, 14))

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    story.append(Paragraph("Interpretation Note", section_style))

    interpretation_table = Table(
        [[
            Paragraph(
                "Prediction confidence is estimated using agreement between RDKit LightGBM "
                "and Hybrid Descriptor + GAT models. Smaller disagreement indicates higher confidence. "
                "This report is intended for research, academic, and portfolio demonstration purposes. "
                "Predictions should be experimentally validated before scientific or industrial use.",
                body_style
            )
        ]],
        colWidths=[502]
    )

    interpretation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), theme_light),
        ("BOX", (0, 0), (-1, -1), 0.8, theme_border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
    ]))

    story.append(interpretation_table)

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

    summary_df_safe = pd.DataFrame(
        summary_data[1:],
        columns=summary_data[0]
    )

    summary_table = safe_pdf_table(
        dataframe=summary_df_safe,
        styles=styles,
        max_columns=3,
        page_width=500
    )

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

    result_table = safe_pdf_table(
        dataframe=display_df,
        styles=styles,
        max_columns=7,
        max_rows=25,
        page_width=500,
        font_size=6,
        header_font_size=6
    )

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

    st.sidebar.success(f"Welcome {get_current_display_name()}")

    st.sidebar.info(
        f"Role: {st.session_state.get('role', 'user')}"
    )

    if st.sidebar.button("Logout"):
        logout_user()


    apply_visible_ui_upgrade()


    render_visible_platform_header()
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

    tab15, tab1, tab2, tab3, tab4, tab6, tab5, tab14, tab8, tab7, tab9, tab10, tab11, tab12, tab13, tab16, tab17, tab18, tab19, tab20 = st.tabs([
        "🏛️ About Platform",
        "🔮 Single Prediction",
        "🧬 Molecule Explorer",
        "📄 Batch CSV",
        "📦 Saved Dataset",
        "📊 Dashboard",
        "🕘 History",
        "🧠 Explainable AI",
        "🛡️ OOD",
        "🧱 Scaffold",
        "📉 PCA",
        "🌀 t-SNE",
        "🌌 UMAP",
        "⚡ Plotly UMAP + AI",
        "💊 Drug-Likeness",
        "👥 Admin Users",
        "🗄️ DataOps / Database + Model Monitoring",
        "📡 Model Monitoring",
        "⚖️ Molecule Comparison",
        "🏆 Scientific Benchmark"
    ])

    with tab1:

        st.subheader("🔮 Single Molecule Prediction")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise prediction workflow: choose molecule → validate structure → select model → generate melting point prediction and report.
            </div>
            """,
            unsafe_allow_html=True
        )

        compound_name = ""
        manual_smiles = ""
        selected_name = "Not Selected"

        # ==================================================
        # INPUT + MODEL CONTROL PANEL
        # ==================================================

        input_card, model_card = st.columns([1.35, 1])

        with input_card:

            with st.container(border=True):

                st.markdown("### 1️⃣ Molecule Input")

                input_mode = st.radio(
                    "Choose Input Method",
                    [
                        "Select from Dataset",
                        "Enter Custom SMILES",
                        "Search by Molecule Name"
                    ],
                    horizontal=True,
                    key="single_prediction_input_mode"
                )

                if input_mode == "Select from Dataset":

                    try:
                        smiles_df = load_molecule_dataset()

                        st.info(
                            "Search or browse the full molecule catalog. The selected molecule is used directly for prediction."
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

                        with st.expander(
                            "Search / Browse Molecule Catalog",
                            expanded=True
                        ):

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
                                    "Reset",
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

                            st.success(
                                f"Matching molecules found: {len(filtered_smiles_df)} out of {len(smiles_df)}"
                            )

                            display_paginated_molecule_table(
                                df=filtered_smiles_df,
                                table_key="single_prediction_catalog",
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

                                manual_smiles = ""
                                selected_name = "Not Selected"

                            else:

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

                                st.session_state["selected_catalog_molecule_name"] = (
                                    selected_name
                                )

                                st.session_state["selected_catalog_smiles"] = (
                                    manual_smiles
                                )

                        if not filtered_smiles_df.empty:

                            selected_name = st.session_state.get(
                                "selected_catalog_molecule_name",
                                filtered_smiles_df.iloc[0]["Molecule_Name"]
                            )

                            manual_smiles = st.session_state.get(
                                "selected_catalog_smiles",
                                filtered_smiles_df.iloc[0]["SMILES"]
                            )

                    except Exception as e:

                        st.error(f"Dataset loading failed: {e}")

                        manual_smiles = st.text_input(
                            "Enter SMILES",
                            value="",
                            key="fallback_single_smiles"
                        )

                        selected_name = "Manual Input"

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

                    if st.button(
                        "Convert Name to SMILES",
                        key="single_name_to_smiles_button"
                    ):

                        resolved_smiles, resolved_name, match_source = resolve_molecule_name_to_smiles(
                            compound_name,
                            load_molecule_dataset()
                        )

                        if resolved_smiles:

                            st.session_state["converted_smiles"] = resolved_smiles
                            st.session_state["compound_name"] = resolved_name
                            st.session_state["name_search_valid"] = True

                            st.success(f"Molecule Name: {resolved_name}")
                            st.success(f"SMILES Found from {match_source}: {resolved_smiles}")

                        else:

                            st.session_state["converted_smiles"] = ""
                            st.session_state["compound_name"] = compound_name
                            st.session_state["name_search_valid"] = False

                            st.error(
                                "Could not find SMILES for this molecule name in the local catalog or PubChem. "
                                "Please select from dataset or enter valid SMILES manually."
                            )

                    manual_smiles = st.session_state.get("converted_smiles", "")
                    selected_name = st.session_state.get("compound_name", compound_name)

                    if manual_smiles:
                        st.success(f"Resolved SMILES: {manual_smiles}")
                    else:
                        st.warning(
                            "Current SMILES: Not available yet. Please convert a valid molecule name first."
                        )

                st.markdown("---")

                st.markdown("### Current Prediction Molecule")

                st.code(
                    f"Molecule Name: {selected_name}\nSMILES: {manual_smiles if manual_smiles else 'Not available'}",
                    language="text"
                )

                if manual_smiles:

                    safe_copy_name_key = make_safe_filename(
                        selected_name
                    )

                    safe_copy_smiles_key = make_safe_filename(
                        manual_smiles
                    )

                    with st.expander("Copy molecule details", expanded=False):

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

        with model_card:

            with st.container(border=True):

                st.markdown("### 2️⃣ Model Selection")

                model_choice = st.radio(
                    "Select Prediction Model",
                    [
                        "RDKit LightGBM",
                        "Hybrid Descriptor + GAT",
                        "Ensemble AI Prediction"
                    ],
                    key="single_prediction_model_choice"
                )

                st.info(
                    "Recommended default: Ensemble AI Prediction. "
                    "It blends descriptor ML and graph-based AI for a stronger final estimate."
                )

                st.markdown("### Model Architecture")

                architecture_df = pd.DataFrame({
                    "Model": [
                        "RDKit LightGBM",
                        "Hybrid Descriptor + GAT",
                        "Ensemble AI"
                    ],
                    "Role": [
                        "Descriptor-based ML",
                        "Graph neural network + descriptors",
                        "Weighted final prediction"
                    ]
                })

                st.dataframe(
                    architecture_df,
                    width="stretch"
                )

        # ==================================================
        # VALIDATION + MOLECULE PREVIEW
        # ==================================================

        if manual_smiles is None or str(manual_smiles).strip() == "":

            st.error(
                "No valid SMILES is available for prediction. "
                "Please convert a valid molecule name, select from dataset, or enter custom SMILES."
            )
            st.stop()

        mol = Chem.MolFromSmiles(
            manual_smiles
        )

        molecule_image = None
        properties_df = None

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 3️⃣ Molecule Validation & Structure")

            if mol is not None:

                preview_col1, preview_col2 = st.columns([1, 1.25])

                with preview_col1:

                    molecule_image = Draw.MolToImage(
                        mol,
                        size=(400, 400)
                    )

                    st.image(
                        molecule_image,
                        caption="2D Molecular Structure"
                    )

                with preview_col2:

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

                    mw_value = float(properties_df.loc[
                        properties_df["Property"] == "Molecular Weight",
                        "Value"
                    ].iloc[0])

                    logp_value = float(properties_df.loc[
                        properties_df["Property"] == "LogP",
                        "Value"
                    ].iloc[0])

                    metric_a, metric_b, metric_c = st.columns(3)

                    with metric_a:
                        st.metric(
                            "Molecular Weight",
                            f"{mw_value:.2f}"
                        )

                    with metric_b:
                        st.metric(
                            "LogP",
                            f"{logp_value:.2f}"
                        )

                    with metric_c:
                        st.metric(
                            "Atoms",
                            Descriptors.HeavyAtomCount(mol)
                        )

                    with st.expander(
                        "View full molecular property table",
                        expanded=True
                    ):

                        properties_df["Value"] = properties_df["Value"].astype(str)

                        st.dataframe(
                            properties_df,
                            width="stretch"
                        )

            else:

                st.error("Invalid SMILES string.")
                st.stop()

        # ==================================================
        # PREDICTION EXECUTION
        # ==================================================

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 4️⃣ Generate Prediction")

            st.write(
                "Click the button below to calculate the melting point and generate confidence, uncertainty, explanation, and report outputs."
            )

            run_prediction = st.button(
                "🚀 Predict Melting Point",
                key="enterprise_single_predict_button",
                type="primary"
            )

        if run_prediction:

            try:

                with st.spinner("Running RDKit, Hybrid GAT, and uncertainty calculations..."):

                    rdkit_prediction = float(
                        predict_melting_point(manual_smiles)
                    )

                    hybrid_prediction = float(
                        predict_hybrid_gat(manual_smiles)
                    )

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

                    model_comparison_df["Prediction_C"] = (
                        model_comparison_df["Prediction_K"] - 273.15
                    ).round(2)

                    uncertainty_results = calculate_prediction_uncertainty(
                        rdkit_prediction,
                        hybrid_prediction
                    )

                    prediction_c = prediction_k - 273.15

                st.success("Prediction completed successfully.")

                with st.container(border=True):

                    st.markdown("### Prediction Result Summary")

                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

                    with result_col1:
                        st.metric(
                            "Melting Point",
                            f"{prediction_k:.2f} K"
                        )

                    with result_col2:
                        st.metric(
                            "Melting Point",
                            f"{prediction_c:.2f} °C"
                        )

                    with result_col3:
                        st.metric(
                            "Confidence",
                            f"{uncertainty_results['confidence']}%"
                        )

                    with result_col4:
                        st.metric(
                            "Uncertainty",
                            f"± {uncertainty_results['uncertainty_range']:.2f} K"
                        )

                    st.markdown("---")

                    model_col, confidence_col = st.columns(2)

                    with model_col:

                        st.markdown("#### Model Comparison")

                        st.dataframe(
                            model_comparison_df,
                            width="stretch"
                        )

                        if model_choice == "Ensemble AI Prediction":

                            st.success(
                                "Final Ensemble Prediction calculated using 40% RDKit LightGBM + 60% Hybrid GAT."
                            )

                    with confidence_col:

                        st.markdown("#### Confidence & Reliability")

                        st.metric(
                            "Model Difference",
                            f"{uncertainty_results['difference']:.2f} K"
                        )

                        if uncertainty_results["confidence"] >= 85:
                            st.success(uncertainty_results["confidence_label"])
                        elif uncertainty_results["confidence"] >= 70:
                            st.warning(uncertainty_results["confidence_label"])
                        else:
                            st.error(uncertainty_results["confidence_label"])

                        st.info(
                            f"Molecule: {selected_name}\n\nSMILES: {manual_smiles}\n\nModel Used: {model_choice}"
                        )

                log_prediction(
                    username=get_current_username(),
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

                save_enterprise_prediction(
                    molecule_name=selected_name,
                    smiles=manual_smiles,
                    prediction_type="Single Prediction",
                    model_used=model_choice,
                    predicted_kelvin=prediction_k,
                    predicted_celsius=prediction_c,
                    confidence=uncertainty_results["confidence"],
                    confidence_label=uncertainty_results["confidence_label"],
                    uncertainty=uncertainty_results["uncertainty_range"],
                    status="Success"
                )

                save_enterprise_user_activity(
                    "Prediction",
                    f"Single prediction generated for {manual_smiles}"
                )

                pdf_shap_df = pd.DataFrame()

                with st.expander(
                    "Advanced Model Explanation",
                    expanded=False
                ):

                    if model_choice == "RDKit LightGBM":

                        st.subheader("RDKit LightGBM SHAP Explanation")

                        pdf_shap_df = explain_prediction(
                            manual_smiles
                        )

                        st.dataframe(
                            pdf_shap_df,
                            width="stretch"
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )

                        ax.barh(
                            pdf_shap_df["Feature"],
                            pdf_shap_df["SHAP_Value"]
                        )

                        ax.set_xlabel("SHAP Value")
                        ax.set_ylabel("Feature")
                        ax.set_title("Top RDKit LightGBM SHAP Contributions")

                        st.pyplot(fig)

                        save_enterprise_shap_outputs(
                            smiles=manual_smiles,
                            model_name=model_choice,
                            shap_df=pdf_shap_df
                        )

                        save_enterprise_user_activity(
                            "SHAP Explanation",
                            f"SHAP explanation generated for {manual_smiles}"
                        )

                    elif model_choice == "Hybrid Descriptor + GAT":

                        st.subheader("Hybrid GAT SHAP Explanation")

                        pdf_shap_df = explain_hybrid_gat_prediction(
                            manual_smiles,
                            top_n=10
                        )

                        st.dataframe(
                            pdf_shap_df,
                            width="stretch"
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )

                        ax.barh(
                            pdf_shap_df["Feature"],
                            pdf_shap_df["SHAP_Value"]
                        )

                        ax.set_xlabel("SHAP Value")
                        ax.set_ylabel("Feature")
                        ax.set_title("Top Hybrid GAT SHAP Contributions")

                        st.pyplot(fig)

                        save_enterprise_shap_outputs(
                            smiles=manual_smiles,
                            model_name=model_choice,
                            shap_df=pdf_shap_df
                        )

                        save_enterprise_user_activity(
                            "SHAP Explanation",
                            f"SHAP explanation generated for {manual_smiles}"
                        )

                        st.subheader("Hybrid GAT Feature Importance")

                        hybrid_importance_df = get_hybrid_feature_importance(
                            top_n=15
                        )

                        st.dataframe(
                            hybrid_importance_df,
                            width="stretch"
                        )

                        fig2, ax2 = plt.subplots(
                            figsize=(8, 5)
                        )

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
                            "For Ensemble AI Prediction, the report includes model comparison, uncertainty, molecular properties, and similar molecules. Use individual models for SHAP-specific PDF sections."
                        )

                with st.expander(
                    "Similar Molecules & PDF Report",
                    expanded=True
                ):

                    full_similarity_df = load_molecule_dataset()

                    pdf_similar_df = find_top_similar_molecules(
                        query_smiles=manual_smiles,
                        molecule_df=full_similarity_df,
                        top_n=10
                    )

                    st.markdown("#### Top Similar Molecules")

                    st.dataframe(
                        pdf_similar_df,
                        width="stretch"
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

                        safe_pdf_name = make_safe_filename(
                            selected_name
                        )

                        st.download_button(
                            label="Download Enhanced Prediction Report PDF",
                            data=pdf_bytes,
                            file_name=f"{safe_pdf_name}_enhanced_prediction_report.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:

                st.error(f"Prediction failed: {e}")

                log_prediction(
                    username=get_current_username(),
                    smiles=manual_smiles,
                    model_used=model_choice,
                    prediction_k=None,
                    prediction_c=None,
                    status=f"Failed: {e}"
                )


    with tab2:

        st.subheader("🧬 Molecule Explorer")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise molecule exploration workflow: search catalog → select molecule → inspect 2D/3D structure → review descriptors → export assets.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            explorer_full_df = load_molecule_dataset()

            # ==================================================
            # SEARCH + SELECTION PANEL
            # ==================================================

            search_panel, summary_panel = st.columns([1.45, 1])

            with search_panel:

                with st.container(border=True):

                    st.markdown("### 1️⃣ Search Molecule Catalog")

                    st.info(
                        "Browse the full molecule catalog. Search by molecule name, partial IUPAC/name, or SMILES."
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

                    col_explorer_search, col_explorer_reset = st.columns([5, 1])

                    with col_explorer_search:

                        explorer_search_query = st.text_input(
                            "Search molecule by IUPAC/name or SMILES",
                            value="",
                            key=explorer_search_key,
                            placeholder="Example: ethanol, benz, acid, FC1=C(F)"
                        )

                    with col_explorer_reset:

                        st.write("")
                        st.write("")

                        if st.button(
                            "Reset",
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

                    st.success(
                        f"Matching molecules found: {len(explorer_df)} out of {len(explorer_full_df)}"
                    )

                    display_paginated_molecule_table(
                        df=explorer_df,
                        table_key="enterprise_explorer_catalog",
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
                        label="Download Filtered Molecule Catalog CSV",
                        data=filtered_explorer_csv,
                        file_name="molecule_explorer_filtered_catalog.csv",
                        mime="text/csv",
                        key="download_explorer_filtered_catalog"
                    )

                    st.markdown("---")

                    if explorer_df.empty:

                        st.warning(
                            "No molecule found. Please search another name or SMILES."
                        )

                        explorer_selected_name = "Not Selected"
                        explorer_selected_smiles = ""

                    else:

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

            with summary_panel:

                with st.container(border=True):

                    st.markdown("### 2️⃣ Selected Molecule")

                    explorer_selected_name = st.session_state.get(
                        "explorer_selected_molecule_name",
                        explorer_df.iloc[0]["Molecule_Name"] if not explorer_df.empty else "Not Selected"
                    )

                    explorer_selected_smiles = st.session_state.get(
                        "explorer_selected_smiles",
                        explorer_df.iloc[0]["SMILES"] if not explorer_df.empty else ""
                    )

                    if explorer_selected_smiles:

                        st.success(f"Molecule: {explorer_selected_name}")
                        st.code(
                            f"SMILES: {explorer_selected_smiles}",
                            language="text"
                        )

                        mol_preview = Chem.MolFromSmiles(
                            explorer_selected_smiles
                        )

                        if mol_preview is not None:

                            col_m1, col_m2 = st.columns(2)

                            with col_m1:
                                st.metric(
                                    "Molecular Weight",
                                    f"{Descriptors.MolWt(mol_preview):.2f}"
                                )

                            with col_m2:
                                st.metric(
                                    "LogP",
                                    f"{Descriptors.MolLogP(mol_preview):.2f}"
                                )

                            col_m3, col_m4 = st.columns(2)

                            with col_m3:
                                st.metric(
                                    "Rings",
                                    Descriptors.RingCount(mol_preview)
                                )

                            with col_m4:
                                st.metric(
                                    "Heavy Atoms",
                                    Descriptors.HeavyAtomCount(mol_preview)
                                )

                    else:

                        st.warning("No molecule selected yet.")

        except Exception as e:

            st.error(f"Molecule Explorer failed to load dataset: {e}")
            explorer_selected_name = "Not Selected"
            explorer_selected_smiles = ""

        if explorer_selected_smiles is None or str(explorer_selected_smiles).strip() == "":
            st.stop()

        # ==================================================
        # STRUCTURE VISUALIZATION
        # ==================================================

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 3️⃣ Molecular Structure Visualization")

            mol = Chem.MolFromSmiles(
                explorer_selected_smiles
            )

            if mol is None:

                st.error("Selected SMILES could not be converted to a valid molecule.")
                st.stop()

            safe_name = make_safe_filename(
                explorer_selected_name
            )

            structure_col_2d, structure_col_3d = st.columns([1, 1.35])

            with structure_col_2d:

                st.markdown("#### 2D Structure")

                molecule_image = Draw.MolToImage(
                    mol,
                    size=(430, 430)
                )

                st.image(
                    molecule_image,
                    caption=f"2D Structure: {explorer_selected_name}"
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

            with structure_col_3d:

                st.markdown("#### Interactive 3D Viewer")

                st.info(
                    "3D visualization uses the existing py3Dmol rendering system. "
                    "No 3D backend logic was changed."
                )

                show_3d_molecule(
                    explorer_selected_smiles,
                    width=650,
                    height=480,
                    viewer_key=f"explorer_3d_{safe_name}"
                )

        # ==================================================
        # DESCRIPTORS + EXPORTS
        # ==================================================

        st.markdown("---")

        descriptor_col, export_col = st.columns([1.2, 1])

        with descriptor_col:

            with st.container(border=True):

                st.markdown("### 4️⃣ Molecular Descriptor Profile")

                descriptor_df = pd.DataFrame({
                    "Descriptor": [
                        "Molecular Formula",
                        "Molecular Weight",
                        "LogP",
                        "TPSA",
                        "H-Bond Donors",
                        "H-Bond Acceptors",
                        "Rotatable Bonds",
                        "Ring Count",
                        "Heavy Atom Count"
                    ],
                    "Value": [
                        rdMolDescriptors.CalcMolFormula(mol),
                        round(Descriptors.MolWt(mol), 2),
                        round(Descriptors.MolLogP(mol), 2),
                        round(Descriptors.TPSA(mol), 2),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.RingCount(mol),
                        Descriptors.HeavyAtomCount(mol)
                    ]
                })

                descriptor_df["Value"] = descriptor_df["Value"].astype(str)

                st.dataframe(
                    descriptor_df,
                    width="stretch"
                )

                descriptor_csv = descriptor_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Descriptor CSV",
                    data=descriptor_csv,
                    file_name=f"{safe_name}_descriptors.csv",
                    mime="text/csv"
                )

        with export_col:

            with st.container(border=True):

                st.markdown("### 5️⃣ Copy / Export Molecule Details")

                st.text_area(
                    "Molecule Name",
                    value=explorer_selected_name,
                    height=70,
                    key=f"explorer_copy_name_{safe_name}"
                )

                st.text_area(
                    "SMILES",
                    value=explorer_selected_smiles,
                    height=90,
                    key=f"explorer_copy_smiles_{safe_name}"
                )

                molecule_details_df = pd.DataFrame([{
                    "Molecule_Name": explorer_selected_name,
                    "SMILES": explorer_selected_smiles
                }])

                molecule_details_csv = molecule_details_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Molecule Details CSV",
                    data=molecule_details_csv,
                    file_name=f"{safe_name}_molecule_details.csv",
                    mime="text/csv"
                )

                st.info(
                    "Use this tab for structure inspection. For prediction, go to 🔮 Single Prediction."
                )


    with tab3:

        st.subheader("📄 Batch CSV Prediction")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise batch workflow: upload CSV → validate SMILES → select row limit → run ensemble predictions → export CSV/PDF report.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==================================================
        # UPLOAD + REQUIREMENTS PANEL
        # ==================================================

        upload_col, guide_col = st.columns([1.25, 1])

        with upload_col:

            with st.container(border=True):

                st.markdown("### 1️⃣ Upload Batch CSV")

                uploaded_file = st.file_uploader(
                    "Upload CSV with a SMILES column",
                    type=["csv"],
                    key="enterprise_batch_csv_upload"
                )

                st.caption(
                    "Required column: SMILES. Optional column: Molecule_Name or Name."
                )

        with guide_col:

            with st.container(border=True):

                st.markdown("### 2️⃣ CSV Format Guide")

                example_batch_df = pd.DataFrame({
                    "Molecule_Name": [
                        "ethanol",
                        "benzene"
                    ],
                    "SMILES": [
                        "CCO",
                        "c1ccccc1"
                    ]
                })

                st.dataframe(
                    example_batch_df,
                    width="stretch"
                )

                example_csv = example_batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Example CSV Template",
                    data=example_csv,
                    file_name="batch_prediction_template.csv",
                    mime="text/csv"
                )

        if uploaded_file is not None:

            df = pd.read_csv(
                uploaded_file
            )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Uploaded Dataset Validation")

                col_v1, col_v2, col_v3 = st.columns(3)

                with col_v1:
                    st.metric(
                        "Uploaded Rows",
                        len(df)
                    )

                with col_v2:
                    st.metric(
                        "Columns",
                        len(df.columns)
                    )

                with col_v3:
                    st.metric(
                        "SMILES Column",
                        "Yes" if "SMILES" in df.columns else "No"
                    )

                display_paginated_dataframe(
                    df=df,
                    table_key="batch_uploaded_preview",
                    rows_per_page=100
                )

                if "SMILES" not in df.columns:

                    st.error(
                        "CSV must contain a column named SMILES."
                    )

                    st.stop()

                smiles_list_all = df["SMILES"].dropna().astype(str).tolist()

                valid_smiles_count = 0
                invalid_smiles_count = 0

                for smiles in smiles_list_all:

                    if Chem.MolFromSmiles(smiles) is not None:
                        valid_smiles_count += 1
                    else:
                        invalid_smiles_count += 1

                col_s1, col_s2, col_s3 = st.columns(3)

                with col_s1:
                    st.metric(
                        "Non-empty SMILES",
                        len(smiles_list_all)
                    )

                with col_s2:
                    st.metric(
                        "Valid RDKit SMILES",
                        valid_smiles_count
                    )

                with col_s3:
                    st.metric(
                        "Invalid SMILES",
                        invalid_smiles_count
                    )

                if len(smiles_list_all) == 0:

                    st.error(
                        "No valid SMILES values found in uploaded CSV."
                    )

                    st.stop()

            # ==================================================
            # BATCH SETTINGS
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 4️⃣ Batch Prediction Settings")

                batch_limit = st.slider(
                    "Number of rows to predict",
                    min_value=1,
                    max_value=len(smiles_list_all),
                    value=min(100, len(smiles_list_all)),
                    step=1,
                    key="enterprise_batch_prediction_limit"
                )

                st.warning(
                    "Cloud safety note: for Streamlit Cloud, test 50–100 molecules first. "
                    "Increase gradually after confirming performance."
                )

                include_invalid_rows = st.checkbox(
                    "Include invalid SMILES rows in output report",
                    value=True,
                    key="enterprise_batch_include_invalid"
                )

                run_batch_prediction = st.button(
                    "🚀 Run Batch Prediction",
                    key="enterprise_run_batch_prediction",
                    type="primary"
                )

            # ==================================================
            # RUN BATCH PREDICTION
            # ==================================================

            if run_batch_prediction:

                prediction_input_df = df.head(
                    batch_limit
                ).copy()

                smiles_list = prediction_input_df[
                    "SMILES"
                ].dropna().astype(str).tolist()

                batch_results = []

                progress_bar = st.progress(
                    0
                )

                status_placeholder = st.empty()

                with st.spinner(
                    f"Running enhanced batch prediction for {len(smiles_list)} molecules..."
                ):

                    for i, smiles in enumerate(smiles_list):

                        status_placeholder.info(
                            f"Processing molecule {i + 1} of {len(smiles_list)}"
                        )

                        mol = Chem.MolFromSmiles(
                            smiles
                        )

                        if mol is None:

                            if include_invalid_rows:

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
                                    "Status": "Invalid SMILES"
                                })

                            progress_bar.progress(
                                int((i + 1) / len(smiles_list) * 100)
                            )

                            continue

                        try:

                            rdkit_pred = float(
                                predict_melting_point(smiles)
                            )

                            hybrid_pred = float(
                                predict_hybrid_gat(smiles)
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

                status_placeholder.empty()

                batch_df = pd.DataFrame(
                    batch_results
                )

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 5️⃣ Batch Prediction Results")

                    success_count = int(
                        (
                            batch_df["Status"] == "Success"
                        ).sum()
                    )

                    failed_count = len(batch_df) - success_count

                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

                    with col_r1:
                        st.metric(
                            "Processed",
                            len(batch_df)
                        )

                    with col_r2:
                        st.metric(
                            "Successful",
                            success_count
                        )

                    with col_r3:
                        st.metric(
                            "Failed / Invalid",
                            failed_count
                        )

                    with col_r4:

                        if success_count > 0:

                            st.metric(
                                "Mean Prediction",
                                f"{batch_df.loc[batch_df['Status'] == 'Success', 'Ensemble_Prediction_K'].mean():.2f} K"
                            )

                        else:

                            st.metric(
                                "Mean Prediction",
                                "N/A"
                            )

                    display_paginated_dataframe(
                        df=batch_df,
                        table_key="enterprise_batch_prediction_results",
                        rows_per_page=100
                    )

                    csv = batch_df.to_csv(
                        index=False
                    ).encode("utf-8")

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

                        pdf_bytes = create_batch_summary_pdf(
                            successful_batch_df
                        )

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

        else:

            st.info(
                "Upload a CSV file to begin enterprise batch prediction."
            )



    with tab4:

        st.subheader("📦 Saved Dataset Prediction")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise saved-dataset workflow: load full dataset → browse with pagination → choose safe prediction limit → generate predictions → export results.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            full_df = pd.read_csv("all_smiles_clean.csv")

            # ==================================================
            # DATASET STATUS PANEL
            # ==================================================

            with st.container(border=True):

                st.markdown("### 1️⃣ Saved Dataset Status")

                col_sd1, col_sd2, col_sd3, col_sd4 = st.columns(4)

                with col_sd1:
                    st.metric(
                        "Rows Loaded",
                        len(full_df)
                    )

                with col_sd2:
                    st.metric(
                        "Columns",
                        len(full_df.columns)
                    )

                with col_sd3:
                    st.metric(
                        "SMILES Column",
                        "Yes" if "SMILES" in full_df.columns else "No"
                    )

                with col_sd4:
                    if "SMILES" in full_df.columns:
                        st.metric(
                            "Unique SMILES",
                            full_df["SMILES"].nunique()
                        )
                    else:
                        st.metric(
                            "Unique SMILES",
                            "N/A"
                        )

                if "SMILES" not in full_df.columns:

                    st.error(
                        "Saved dataset must contain a SMILES column."
                    )

                    st.stop()

                st.success(
                    f"Saved full descriptor dataset loaded successfully with {len(full_df)} rows."
                )

            # ==================================================
            # DATASET BROWSER
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 2️⃣ Browse Saved Dataset")

                st.info(
                    "Use pagination to browse the complete saved dataset. "
                    "This preview does not run predictions."
                )

                if "saved_dataset_search_reset_counter" not in st.session_state:
                    st.session_state["saved_dataset_search_reset_counter"] = 0

                saved_dataset_search_key = (
                    "saved_dataset_search_"
                    + str(st.session_state["saved_dataset_search_reset_counter"])
                )

                saved_search_col, saved_reset_col = st.columns([5, 1])

                with saved_search_col:

                    search_saved_dataset = st.text_input(
                        "Search saved dataset by SMILES or molecule name if available",
                        value="",
                        key=saved_dataset_search_key
                    )

                with saved_reset_col:

                    st.write("")
                    st.write("")

                    if st.button(
                        "🔄 Reset",
                        key="reset_saved_dataset_search_button"
                    ):

                        st.session_state["saved_dataset_search_reset_counter"] += 1

                        for pagination_key in [
                            "enterprise_saved_dataset_preview_page_number",
                            "enterprise_saved_dataset_preview_rows_per_page"
                        ]:
                            if pagination_key in st.session_state:
                                del st.session_state[pagination_key]

                        st.rerun()


                if search_saved_dataset.strip() != "":

                    searchable_cols = [
                        col for col in full_df.columns
                        if full_df[col].dtype == "object"
                    ]

                    if searchable_cols:

                        search_mask = False

                        for col in searchable_cols:

                            col_mask = full_df[col].astype(str).str.contains(
                                search_saved_dataset,
                                case=False,
                                na=False,
                                regex=False
                            )

                            if isinstance(search_mask, bool):
                                search_mask = col_mask
                            else:
                                search_mask = search_mask | col_mask

                        preview_df = full_df[
                            search_mask
                        ].copy()

                    else:

                        preview_df = full_df.copy()

                else:

                    preview_df = full_df.copy()

                st.success(
                    f"Matching rows: {len(preview_df)} out of {len(full_df)}"
                )

                display_paginated_dataframe(
                    df=preview_df,
                    table_key="enterprise_saved_dataset_preview",
                    rows_per_page=100
                )

                preview_csv = preview_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Current Filtered Dataset CSV",
                    data=preview_csv,
                    file_name="saved_dataset_filtered_preview.csv",
                    mime="text/csv",
                    key="download_saved_dataset_filtered_preview_csv"
                )

            # ==================================================
            # PREDICTION SETTINGS
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Prediction Settings")

                valid_smiles_df = full_df[
                    full_df["SMILES"].notna()
                ].copy()

                valid_smiles_df["SMILES"] = valid_smiles_df[
                    "SMILES"
                ].astype(str)

                valid_smiles_df = valid_smiles_df[
                    valid_smiles_df["SMILES"].str.strip() != ""
                ].copy()

                col_ps1, col_ps2, col_ps3 = st.columns(3)

                with col_ps1:
                    st.metric(
                        "Available SMILES Rows",
                        len(valid_smiles_df)
                    )

                with col_ps2:
                    st.metric(
                        "Default Safe Limit",
                        min(100, len(valid_smiles_df))
                    )

                with col_ps3:
                    st.metric(
                        "Cloud Mode",
                        "Safe"
                    )

                max_prediction_rows = st.slider(
                    "Number of rows to predict",
                    min_value=10,
                    max_value=len(valid_smiles_df),
                    value=min(100, len(valid_smiles_df)),
                    step=10,
                    key="saved_full_dataset_prediction_limit"
                )

                st.warning(
                    "Cloud safety note: test 100 rows first, then 500, then larger batches. "
                    "Full 3328-row prediction may be heavy on Streamlit Cloud."
                )

                prediction_scope = st.radio(
                    "Prediction source",
                    [
                        "First N rows from full dataset",
                        "First N rows from current filtered search"
                    ],
                    horizontal=True,
                    key="saved_dataset_prediction_scope"
                )

                run_saved_prediction = st.button(
                    "🚀 Run Saved Dataset Prediction",
                    key="run_saved_full_dataset_prediction",
                    type="primary"
                )

            # ==================================================
            # RUN PREDICTIONS
            # ==================================================

            if run_saved_prediction:

                if prediction_scope == "First N rows from current filtered search":

                    source_df = preview_df.copy()

                    if "SMILES" not in source_df.columns:
                        st.error(
                            "Filtered dataset does not contain SMILES column."
                        )
                        st.stop()

                    source_df = source_df[
                        source_df["SMILES"].notna()
                    ].copy()

                    source_df["SMILES"] = source_df["SMILES"].astype(str)

                    source_df = source_df[
                        source_df["SMILES"].str.strip() != ""
                    ].copy()

                else:

                    source_df = valid_smiles_df.copy()

                prediction_df = source_df.head(
                    max_prediction_rows
                ).copy()

                prediction_results = []

                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                with st.spinner(
                    f"Running prediction on {len(prediction_df)} saved-dataset rows..."
                ):

                    for i, (_, row) in enumerate(prediction_df.iterrows()):

                        smiles = str(row["SMILES"]).strip()

                        status_placeholder.info(
                            f"Processing row {i + 1} of {len(prediction_df)}"
                        )

                        mol = Chem.MolFromSmiles(
                            smiles
                        )

                        if mol is None:

                            prediction_results.append({
                                "SMILES": smiles,
                                "RDKit_Prediction_K": None,
                                "Hybrid_GAT_Prediction_K": None,
                                "Ensemble_Prediction_K": None,
                                "Ensemble_Prediction_C": None,
                                "Model_Difference_K": None,
                                "Estimated_Uncertainty_K": None,
                                "Confidence_%": None,
                                "Confidence_Label": None,
                                "Status": "Invalid SMILES"
                            })

                            progress_bar.progress(
                                int((i + 1) / len(prediction_df) * 100)
                            )

                            continue

                        try:

                            rdkit_prediction = float(
                                predict_melting_point(smiles)
                            )

                            hybrid_prediction = float(
                                predict_hybrid_gat(smiles)
                            )

                            ensemble_prediction = (
                                0.4 * rdkit_prediction
                                +
                                0.6 * hybrid_prediction
                            )

                            uncertainty = calculate_prediction_uncertainty(
                                rdkit_prediction,
                                hybrid_prediction
                            )

                            prediction_results.append({
                                "SMILES": smiles,
                                "RDKit_Prediction_K": round(rdkit_prediction, 2),
                                "Hybrid_GAT_Prediction_K": round(hybrid_prediction, 2),
                                "Ensemble_Prediction_K": round(ensemble_prediction, 2),
                                "Ensemble_Prediction_C": round(ensemble_prediction - 273.15, 2),
                                "Model_Difference_K": uncertainty["difference"],
                                "Estimated_Uncertainty_K": uncertainty["uncertainty_range"],
                                "Confidence_%": uncertainty["confidence"],
                                "Confidence_Label": uncertainty["confidence_label"],
                                "Status": "Success"
                            })

                        except Exception as row_error:

                            prediction_results.append({
                                "SMILES": smiles,
                                "RDKit_Prediction_K": None,
                                "Hybrid_GAT_Prediction_K": None,
                                "Ensemble_Prediction_K": None,
                                "Ensemble_Prediction_C": None,
                                "Model_Difference_K": None,
                                "Estimated_Uncertainty_K": None,
                                "Confidence_%": None,
                                "Confidence_Label": None,
                                "Status": f"Failed: {row_error}"
                            })

                        progress_bar.progress(
                            int((i + 1) / len(prediction_df) * 100)
                        )

                status_placeholder.empty()

                results_df = pd.DataFrame(
                    prediction_results
                )

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 4️⃣ Saved Dataset Prediction Results")

                    success_count = int(
                        (
                            results_df["Status"] == "Success"
                        ).sum()
                    )

                    failed_count = len(results_df) - success_count

                    col_rs1, col_rs2, col_rs3, col_rs4 = st.columns(4)

                    with col_rs1:
                        st.metric(
                            "Processed",
                            len(results_df)
                        )

                    with col_rs2:
                        st.metric(
                            "Successful",
                            success_count
                        )

                    with col_rs3:
                        st.metric(
                            "Failed / Invalid",
                            failed_count
                        )

                    with col_rs4:

                        if success_count > 0:

                            st.metric(
                                "Mean Ensemble K",
                                f"{results_df.loc[results_df['Status'] == 'Success', 'Ensemble_Prediction_K'].mean():.2f}"
                            )

                        else:

                            st.metric(
                                "Mean Ensemble K",
                                "N/A"
                            )

                    display_paginated_dataframe(
                        df=results_df,
                        table_key="enterprise_saved_dataset_prediction_results",
                        rows_per_page=100
                    )

                    result_csv = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Saved Dataset Predictions CSV",
                        data=result_csv,
                        file_name="saved_full_dataset_predictions.csv",
                        mime="text/csv"
                    )

                    successful_results_df = results_df[
                        results_df["Status"] == "Success"
                    ].copy()

                    if not successful_results_df.empty:

                        try:

                            pdf_bytes = create_batch_summary_pdf(
                                successful_results_df.rename(
                                    columns={
                                        "RDKit_Prediction_K": "RDKit_LightGBM_K",
                                        "Hybrid_GAT_Prediction_K": "Hybrid_GAT_K"
                                    }
                                )
                            )

                            st.download_button(
                                label="Download Saved Dataset Summary PDF",
                                data=pdf_bytes,
                                file_name="saved_dataset_prediction_summary.pdf",
                                mime="application/pdf"
                            )

                        except Exception as pdf_error:

                            st.warning(
                                f"PDF summary could not be generated: {pdf_error}"
                            )

        except Exception as e:

            st.error(f"Saved full dataset prediction failed: {e}")


    with tab5:

        st.subheader("🕘 Prediction History")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise audit trail: review prediction logs, filter history, export records, and safely remove selected entries.
            </div>
            """,
            unsafe_allow_html=True
        )

        all_rows = load_prediction_logs()
        rows = filter_logs_for_current_user(all_rows)

        with st.container(border=True):
            st.markdown("### 🔐 User History Scope")
            st.info(get_user_history_scope_label())

        if len(rows) == 0:

            with st.container(border=True):

                st.markdown("### Prediction History Status")
                st.info("No prediction history available yet.")

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

            history_df["Prediction K"] = pd.to_numeric(
                history_df["Prediction K"],
                errors="coerce"
            )

            history_df["Prediction °C"] = pd.to_numeric(
                history_df["Prediction °C"],
                errors="coerce"
            )

            history_df["Prediction Result"] = history_df["Status"].apply(
                lambda x: "Success" if str(x).startswith("Success") else "Failed"
            )

            # ==================================================
            # HISTORY SUMMARY
            # ==================================================

            with st.container(border=True):

                st.markdown("### 1️⃣ Prediction History Summary")

                total_records = len(history_df)

                success_records = int(
                    (
                        history_df["Prediction Result"] == "Success"
                    ).sum()
                )

                failed_records = total_records - success_records

                unique_models = history_df[
                    "Model Used"
                ].nunique()

                col_h1, col_h2, col_h3, col_h4 = st.columns(4)

                with col_h1:
                    st.metric(
                        "Total Records",
                        total_records
                    )

                with col_h2:
                    st.metric(
                        "Successful",
                        success_records
                    )

                with col_h3:
                    st.metric(
                        "Failed",
                        failed_records
                    )

                with col_h4:
                    st.metric(
                        "Models Used",
                        unique_models
                    )

            # ==================================================
            # FILTERS
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 2️⃣ Filter Prediction History")

                if is_admin_user():

                    user_options = [
                        "All"
                    ] + sorted(
                        history_df["Username"].dropna().astype(str).unique().tolist()
                    )

                    selected_user_filter = st.selectbox(
                        "Admin User Filter",
                        options=user_options,
                        key="history_admin_user_filter"
                    )

                    if selected_user_filter != "All":
                        history_df = history_df[
                            history_df["Username"].astype(str) == selected_user_filter
                        ].copy()

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:

                    selected_result_filter = st.selectbox(
                        "Prediction Result",
                        options=[
                            "All",
                            "Success",
                            "Failed"
                        ],
                        key="history_result_filter"
                    )

                with filter_col2:

                    model_options = [
                        "All"
                    ] + sorted(
                        history_df["Model Used"].dropna().astype(str).unique().tolist()
                    )

                    selected_model_filter = st.selectbox(
                        "Model Used",
                        options=model_options,
                        key="history_model_filter"
                    )

                with filter_col3:

                    smiles_search = st.text_input(
                        "Search SMILES",
                        value="",
                        key="history_smiles_search"
                    )

                filtered_history_df = history_df.copy()

                if selected_result_filter != "All":

                    filtered_history_df = filtered_history_df[
                        filtered_history_df["Prediction Result"] == selected_result_filter
                    ].copy()

                if selected_model_filter != "All":

                    filtered_history_df = filtered_history_df[
                        filtered_history_df["Model Used"].astype(str) == selected_model_filter
                    ].copy()

                if smiles_search.strip() != "":

                    filtered_history_df = filtered_history_df[
                        filtered_history_df["SMILES"].astype(str).str.contains(
                            smiles_search,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ].copy()

                st.success(
                    f"Showing {len(filtered_history_df)} of {len(history_df)} history records."
                )

            # ==================================================
            # HISTORY TABLE
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Prediction History Table")

                display_paginated_dataframe(
                    df=filtered_history_df,
                    table_key="enterprise_prediction_history_table",
                    rows_per_page=100
                )

                csv = filtered_history_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Filtered History CSV",
                    data=csv,
                    file_name="prediction_history_filtered.csv",
                    mime="text/csv"
                )

                full_csv = history_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Full History CSV",
                    data=full_csv,
                    file_name="prediction_history_full.csv",
                    mime="text/csv"
                )

            # ==================================================
            # SAFE DELETE / CLEAR
            # ==================================================

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Admin Monitoring Dashboard")

                monitor_users_df, monitor_history_df = build_admin_monitoring_data()

                mon_col1, mon_col2, mon_col3, mon_col4 = st.columns(4)

                total_predictions = len(monitor_history_df)

                successful_predictions = (
                    int(
                        (
                            monitor_history_df["Prediction Result"] == "Success"
                        ).sum()
                    )
                    if not monitor_history_df.empty
                    else 0
                )

                failed_predictions = (
                    int(
                        (
                            monitor_history_df["Prediction Result"] == "Failed"
                        ).sum()
                    )
                    if not monitor_history_df.empty
                    else 0
                )

                active_users = (
                    monitor_history_df["Username"].nunique()
                    if not monitor_history_df.empty
                    else 0
                )

                with mon_col1:
                    st.metric(
                        "Total Predictions",
                        total_predictions
                    )

                with mon_col2:
                    st.metric(
                        "Successful",
                        successful_predictions
                    )

                with mon_col3:
                    st.metric(
                        "Failed",
                        failed_predictions
                    )

                with mon_col4:
                    st.metric(
                        "Active Users",
                        active_users
                    )

                if monitor_history_df.empty:

                    st.info(
                        "No prediction activity available yet."
                    )

                else:

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        st.markdown("#### Predictions by User")

                        predictions_by_user = (
                            monitor_history_df
                            .groupby("Username")
                            .size()
                            .reset_index(name="Prediction Count")
                            .sort_values(
                                by="Prediction Count",
                                ascending=False
                            )
                        )

                        st.bar_chart(
                            predictions_by_user.set_index("Username")
                        )

                    with chart_col2:

                        st.markdown("#### Prediction Result Summary")

                        result_summary = (
                            monitor_history_df
                            .groupby("Prediction Result")
                            .size()
                            .reset_index(name="Count")
                        )

                        st.bar_chart(
                            result_summary.set_index("Prediction Result")
                        )

                    st.markdown("#### Most Active Users")

                    most_active_users = (
                        monitor_history_df
                        .groupby("Username")
                        .agg(
                            Total_Predictions=("ID", "count"),
                            Successful=(
                                "Prediction Result",
                                lambda x: int((x == "Success").sum())
                            ),
                            Failed=(
                                "Prediction Result",
                                lambda x: int((x == "Failed").sum())
                            ),
                            Last_Activity=("Created At", "max")
                        )
                        .reset_index()
                        .sort_values(
                            by="Total_Predictions",
                            ascending=False
                        )
                    )

                    st.dataframe(
                        most_active_users,
                        width="stretch",
                        hide_index=True
                    )

                    admin_summary_report = create_admin_monitoring_report(
                        monitor_users_df,
                        monitor_history_df
                    )

                    admin_summary_csv = admin_summary_report.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Admin Monitoring Summary CSV",
                        data=admin_summary_csv,
                        file_name="admin_monitoring_summary.csv",
                        mime="text/csv"
                    )

                    user_activity_csv = most_active_users.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download User Activity CSV",
                        data=user_activity_csv,
                        file_name="admin_user_activity.csv",
                        mime="text/csv"
                    )

            st.markdown("---")

            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:

                with st.container(border=True):

                    st.markdown("### 4️⃣ Delete Selected Prediction")

                    if filtered_history_df.empty:

                        st.info("No filtered records available to delete.")

                    else:

                        selected_id = st.selectbox(
                            "Select Prediction ID to Delete",
                            filtered_history_df["ID"].tolist(),
                            key="enterprise_history_delete_id"
                        )

                        confirm_delete = st.checkbox(
                            "I confirm deleting the selected prediction row",
                            key="confirm_delete_prediction_row"
                        )

                        if st.button(
                            "Delete Selected Row",
                            key="enterprise_delete_selected_history_row"
                        ):

                            if confirm_delete:

                                delete_prediction_row(
                                    selected_id
                                )

                                st.success(
                                    f"Prediction row {selected_id} deleted."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "Please confirm before deleting the selected row."
                                )

            with action_col2:

                with st.container(border=True):

                    st.markdown("### 5️⃣ Clear Entire History")

                    st.warning(
                        "Admin users clear all stored prediction logs. Regular users should delete only their own filtered rows."
                    )

                    confirm_clear_history = st.checkbox(
                        "I confirm clearing the entire prediction history",
                        key="confirm_clear_prediction_history"
                    )

                    if st.button(
                        "Clear Entire History",
                        key="enterprise_clear_entire_history"
                    ):

                        if confirm_clear_history:

                            if is_admin_user():

                                clear_prediction_logs()

                                st.success(
                                    "All prediction history deleted by admin."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "Only admin users can clear the entire database. Please delete your own selected rows individually."
                                )

                        else:

                            st.warning(
                                "Please confirm before clearing the entire history."
                            )


    with tab6:

        st.subheader("📊 Dashboard Summary")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise dashboard workflow: analyze global prediction activity or generate a detailed dashboard for one selected molecule.
            </div>
            """,
            unsafe_allow_html=True
        )

        dashboard_mode = st.radio(
            "Choose Dashboard Mode",
            [
                "Global Dashboard",
                "Current Molecule Dashboard"
            ],
            horizontal=True,
            key="dashboard_mode_selector"
        )

        # ==================================================
        # GLOBAL DASHBOARD
        # ==================================================

        if dashboard_mode == "Global Dashboard":

            all_rows = load_prediction_logs()
            rows = filter_logs_for_current_user(all_rows)

            st.info(get_user_history_scope_label())

            if len(rows) == 0:

                with st.container(border=True):

                    st.markdown("### Global Dashboard Status")
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
                    dashboard_df[
                        dashboard_df["Prediction Result"] == "Success"
                    ]
                )

                failed_count = len(
                    dashboard_df[
                        dashboard_df["Prediction Result"] == "Failed"
                    ]
                )

                avg_prediction_k = dashboard_df[
                    "Prediction K"
                ].mean()

                avg_prediction_c = dashboard_df[
                    "Prediction °C"
                ].mean()

                avg_confidence = dashboard_df[
                    "Confidence %"
                ].mean()

                with st.container(border=True):

                    st.markdown("### 1️⃣ Global Prediction KPI Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Total Predictions",
                            total_predictions
                        )

                    with col2:
                        st.metric(
                            "Successful",
                            success_count
                        )

                    with col3:
                        st.metric(
                            "Failed",
                            failed_count
                        )

                    with col4:
                        if pd.notna(avg_confidence):
                            st.metric(
                                "Average Confidence",
                                f"{avg_confidence:.2f}%"
                            )
                        else:
                            st.metric(
                                "Average Confidence",
                                "N/A"
                            )

                    col5, col6 = st.columns(2)

                    with col5:
                        if pd.notna(avg_prediction_k):
                            st.metric(
                                "Average Melting Point (K)",
                                f"{avg_prediction_k:.2f} K"
                            )
                        else:
                            st.metric(
                                "Average Melting Point (K)",
                                "N/A"
                            )

                    with col6:
                        if pd.notna(avg_prediction_c):
                            st.metric(
                                "Average Melting Point (°C)",
                                f"{avg_prediction_c:.2f} °C"
                            )
                        else:
                            st.metric(
                                "Average Melting Point (°C)",
                                "N/A"
                            )

                st.markdown("---")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:

                    with st.container(border=True):

                        st.markdown("### 2️⃣ Success vs Failed Predictions")

                        result_counts = dashboard_df[
                            "Prediction Result"
                        ].value_counts()

                        fig_result, ax_result = plt.subplots(
                            figsize=(6, 4)
                        )

                        ax_result.bar(
                            result_counts.index,
                            result_counts.values
                        )

                        ax_result.set_xlabel("Prediction Result")
                        ax_result.set_ylabel("Count")
                        ax_result.set_title("Success vs Failed Predictions")

                        st.pyplot(fig_result)

                with chart_col2:

                    with st.container(border=True):

                        st.markdown("### 3️⃣ Model Usage Count")

                        model_counts = dashboard_df[
                            "Model Used"
                        ].value_counts()

                        fig_model, ax_model = plt.subplots(
                            figsize=(7, 4)
                        )

                        ax_model.bar(
                            model_counts.index,
                            model_counts.values
                        )

                        ax_model.set_xlabel("Model")
                        ax_model.set_ylabel("Count")
                        ax_model.set_title("Model Usage Count")
                        ax_model.tick_params(axis="x", rotation=20)

                        st.pyplot(fig_model)

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 4️⃣ Confidence Distribution")

                    confidence_df = dashboard_df.dropna(
                        subset=[
                            "Confidence %"
                        ]
                    )

                    if confidence_df.empty:

                        st.info(
                            "No confidence values available yet. Run predictions using uncertainty-enabled models."
                        )

                    else:

                        fig_conf, ax_conf = plt.subplots(
                            figsize=(8, 4)
                        )

                        ax_conf.hist(
                            confidence_df["Confidence %"],
                            bins=10
                        )

                        ax_conf.set_xlabel("Confidence %")
                        ax_conf.set_ylabel("Frequency")
                        ax_conf.set_title("Prediction Confidence Distribution")

                        st.pyplot(fig_conf)

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 5️⃣ Recent / Full Dashboard Data")

                    recent_df = dashboard_df.sort_values(
                        by="ID",
                        ascending=False
                    ).head(10)

                    with st.expander(
                        "View 10 Most Recent Predictions",
                        expanded=True
                    ):

                        st.dataframe(
                            recent_df,
                            width="stretch"
                        )

                    with st.expander(
                        "View Full Dashboard Data with Pagination",
                        expanded=False
                    ):

                        display_paginated_dataframe(
                            df=dashboard_df,
                            table_key="enterprise_global_dashboard_data",
                            rows_per_page=100
                        )

                    dashboard_csv = dashboard_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Global Dashboard Data CSV",
                        data=dashboard_csv,
                        file_name="global_dashboard_summary_data.csv",
                        mime="text/csv"
                    )

        # ==================================================
        # CURRENT MOLECULE DASHBOARD
        # ==================================================

        else:

            try:

                current_df = load_molecule_dataset()

                with st.container(border=True):

                    st.markdown("### 1️⃣ Select Current Molecule")

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
                            "Search or browse the molecule catalog, then select a molecule for the dashboard."
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
                                "Reset",
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

                        st.success(
                            f"Matching molecules found: {len(current_filtered_df)} out of {len(current_df)}"
                        )

                        display_paginated_molecule_table(
                            df=current_filtered_df,
                            table_key="enterprise_current_dashboard_catalog",
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

                        if current_filtered_df.empty:

                            st.warning(
                                "No molecule found. Please try another search."
                            )

                            st.stop()

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

                    else:

                        current_name = "Custom Input"

                        current_smiles = st.text_input(
                            "Enter custom SMILES",
                            value="CCO",
                            key="current_dashboard_custom_smiles"
                        )

                    st.markdown("---")

                    st.markdown("### Selected Molecule")

                    st.code(
                        f"Molecule Name: {current_name}\nSMILES: {current_smiles}",
                        language="text"
                    )

                    with st.expander(
                        "Copy selected molecule details",
                        expanded=False
                    ):

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

                    generate_dashboard = st.button(
                        "🚀 Generate Current Molecule Dashboard",
                        key="generate_current_molecule_dashboard",
                        type="primary"
                    )

                if generate_dashboard:

                    mol = Chem.MolFromSmiles(
                        current_smiles
                    )

                    if mol is None:

                        st.error(
                            "Invalid SMILES. Please enter a valid molecule."
                        )

                    else:

                        with st.spinner(
                            "Generating current molecule dashboard..."
                        ):

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

                            save_enterprise_ood_result(
                                smiles=current_smiles,
                                ood_result=ood_result
                            )

                            scaffold_smiles = get_murcko_scaffold(
                                current_smiles
                            )

                            similar_df = find_top_similar_molecules(
                                query_smiles=current_smiles,
                                molecule_df=current_df,
                                top_n=10
                            )

                            ensemble_uncertainty = calculate_deep_ensemble_uncertainty(
                                rdkit_pred,
                                hybrid_pred,
                                ensemble_pred
                            ) if "calculate_deep_ensemble_uncertainty" in globals() else None

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 2️⃣ Current Molecule Prediction Summary")

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

                        with st.container(border=True):

                            st.markdown("### 3️⃣ Molecular Structure Comparison")

                            molecule_image = Draw.MolToImage(
                                mol,
                                size=(350, 350)
                            )

                            col_2d, col_3d = st.columns(2)

                            with col_2d:

                                st.markdown("#### 2D Molecular Structure")

                                st.image(
                                    molecule_image,
                                    caption="Current Molecule 2D Structure"
                                )

                            with col_3d:

                                st.markdown("#### 3D Molecular Structure")

                                st.info(
                                    "Interactive 3D structure is generated using the existing 3D viewer code."
                                )

                                show_3d_molecule(
                                    current_smiles,
                                    width=430,
                                    height=400,
                                    viewer_key="current_dashboard_3d"
                                )

                        st.markdown("---")

                        info_col1, info_col2 = st.columns(2)

                        with info_col1:

                            with st.container(border=True):

                                st.markdown("### 4️⃣ Molecular Properties")

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

                                properties_df["Value"] = properties_df[
                                    "Value"
                                ].astype(str)

                                st.dataframe(
                                    properties_df,
                                    width="stretch"
                                )

                        with info_col2:

                            with st.container(border=True):

                                st.markdown("### 5️⃣ Scaffold Summary")

                                st.code(
                                    f"Murcko Scaffold: {scaffold_smiles}",
                                    language="text"
                                )

                                if scaffold_smiles not in [
                                    None,
                                    "No Scaffold"
                                ]:

                                    scaffold_mol = Chem.MolFromSmiles(
                                        scaffold_smiles
                                    )

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

                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:

                            with st.container(border=True):

                                st.markdown("### 6️⃣ OOD and Nearest Molecule Details")

                                ood_display_df = pd.DataFrame([
                                    ood_result
                                ])

                                st.dataframe(
                                    ood_display_df,
                                    width="stretch"
                                )

                                nearest_smiles = ood_result[
                                    "Nearest_SMILES"
                                ]

                                nearest_name = ood_result[
                                    "Nearest_Molecule_Name"
                                ]

                                if nearest_smiles is not None:

                                    nearest_mol = Chem.MolFromSmiles(
                                        nearest_smiles
                                    )

                                    if nearest_mol is not None:

                                        nearest_img = Draw.MolToImage(
                                            nearest_mol,
                                            size=(300, 300)
                                        )

                                        st.image(
                                            nearest_img,
                                            caption=f"Nearest Dataset Molecule: {nearest_name}"
                                        )

                        with detail_col2:

                            with st.container(border=True):

                                st.markdown("### 7️⃣ Top Similar Molecules")

                                if similar_df.empty:

                                    st.warning(
                                        "No similar molecules found."
                                    )

                                else:

                                    st.dataframe(
                                        similar_df,
                                        width="stretch"
                                    )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 8️⃣ Prediction Explanation & Export")

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

                            st.info(
                                explanation_text
                            )

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

                st.error(
                    f"Current molecule dashboard failed: {e}"
                )


    with tab7:

        st.subheader("🧱 Scaffold Analysis")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise scaffold workflow: select molecule → extract Murcko scaffold → inspect scaffold family → summarize dataset scaffold diversity.
            </div>
            """,
            unsafe_allow_html=True
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

                with st.container(border=True):

                    st.markdown("### 1️⃣ Select Molecule for Scaffold Analysis")

                    st.info(
                        "Search or browse the molecule catalog, then select a molecule to view its Murcko scaffold and scaffold family."
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
                            "Reset",
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

                    st.success(
                        f"Matching molecules found: {len(scaffold_filtered_df)} out of {len(molecule_df)}"
                    )

                    display_paginated_molecule_table(
                        df=scaffold_filtered_df,
                        table_key="enterprise_scaffold_catalog",
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

                    if scaffold_filtered_df.empty:

                        st.warning(
                            "No molecule found for this search. Please try another molecule name or SMILES."
                        )

                        st.stop()

                    scaffold_selected_index = st.selectbox(
                        "Choose molecule from filtered list",
                        options=scaffold_filtered_df.index,
                        format_func=lambda x: scaffold_filtered_df.loc[
                            x,
                            "Molecule_Display"
                        ],
                        key=scaffold_selectbox_key
                    )

                    selected_scaffold_name = scaffold_filtered_df.loc[
                        scaffold_selected_index,
                        "Molecule_Name"
                    ]

                    selected_scaffold_smiles = scaffold_filtered_df.loc[
                        scaffold_selected_index,
                        "SMILES"
                    ]

                    st.session_state["scaffold_selected_molecule_name"] = (
                        selected_scaffold_name
                    )

                    st.session_state["scaffold_selected_smiles"] = (
                        selected_scaffold_smiles
                    )

                    st.markdown("---")

                    st.markdown("### Selected Molecule")

                    st.code(
                        f"Molecule Name: {selected_scaffold_name}\nSMILES: {selected_scaffold_smiles}",
                        language="text"
                    )

                    with st.expander(
                        "Copy selected molecule details",
                        expanded=False
                    ):

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

                selected_mol = Chem.MolFromSmiles(
                    selected_scaffold_smiles
                )

                if selected_mol is None:

                    st.error(
                        "Invalid SMILES for selected molecule."
                    )

                    st.stop()

                selected_scaffold_smiles_core = get_murcko_scaffold(
                    selected_scaffold_smiles
                )

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 2️⃣ Molecule Structure vs Murcko Scaffold")

                    col_molecule, col_scaffold = st.columns(2)

                    with col_molecule:

                        st.markdown("#### Selected Molecule 2D Structure")

                        selected_molecule_image = Draw.MolToImage(
                            selected_mol,
                            size=(350, 350)
                        )

                        st.image(
                            selected_molecule_image,
                            caption="Selected Molecule"
                        )

                    with col_scaffold:

                        st.markdown("#### Murcko Scaffold 2D Structure")

                        if selected_scaffold_smiles_core in [
                            None,
                            "No Scaffold"
                        ]:

                            st.warning(
                                "No Murcko scaffold found for this molecule. This often happens for acyclic molecules."
                            )

                        else:

                            scaffold_mol = Chem.MolFromSmiles(
                                selected_scaffold_smiles_core
                            )

                            if scaffold_mol is None:

                                st.warning(
                                    "Scaffold structure could not be rendered."
                                )

                            else:

                                scaffold_image = Draw.MolToImage(
                                    scaffold_mol,
                                    size=(350, 350)
                                )

                                st.image(
                                    scaffold_image,
                                    caption="Murcko Scaffold Core"
                                )

                    st.markdown("#### Murcko Scaffold SMILES")

                    st.code(
                        f"{selected_scaffold_smiles_core}",
                        language="text"
                    )

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 3️⃣ Molecules Sharing the Same Scaffold")

                    with st.spinner(
                        "Generating scaffold family from dataset..."
                    ):

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

                        display_paginated_dataframe(
                            df=same_scaffold_display_df,
                            table_key="enterprise_same_scaffold_family",
                            rows_per_page=100
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

                with st.container(border=True):

                    st.markdown("### 4️⃣ Scaffold Family Summary")

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

                with st.container(border=True):

                    st.markdown("### 1️⃣ Full Dataset Scaffold Settings")

                    st.info(
                        "Generate Murcko scaffold statistics and scaffold frequency analysis. Use row limits on Streamlit Cloud for safety, then increase gradually."
                    )

                    scaffold_process_limit = st.slider(
                        "Number of molecules to process for scaffold summary",
                        min_value=50,
                        max_value=len(molecule_df),
                        value=min(500, len(molecule_df)),
                        step=50,
                        key="full_scaffold_process_limit"
                    )

                    st.warning(
                        "Cloud safety note: start with 500 molecules. If stable, increase to 1000, 2000, and then the full dataset."
                    )

                    run_full_scaffold = st.button(
                        "🚀 Generate Full Dataset Scaffold Analysis",
                        key="generate_full_scaffold_analysis",
                        type="primary"
                    )

                if run_full_scaffold:

                    with st.spinner(
                        f"Generating Murcko scaffolds for {scaffold_process_limit} molecules..."
                    ):

                        scaffold_input_df = molecule_df.head(
                            scaffold_process_limit
                        ).copy()

                        scaffold_df = generate_scaffold_dataframe(
                            scaffold_input_df
                        )

                    if scaffold_df.empty:

                        st.warning(
                            "No valid scaffolds could be generated."
                        )

                    else:

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

                        total_molecules = len(
                            scaffold_df
                        )

                        unique_scaffolds = scaffold_df[
                            "Murcko_Scaffold"
                        ].nunique()

                        no_scaffold_count = len(
                            scaffold_df[
                                scaffold_df["Murcko_Scaffold"] == "No Scaffold"
                            ]
                        )

                        with st.container(border=True):

                            st.markdown("### 2️⃣ Scaffold KPI Summary")

                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

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

                            with col_s4:
                                st.metric(
                                    "Processed Limit",
                                    scaffold_process_limit
                                )

                        st.markdown("---")

                        table_col1, table_col2 = st.columns(2)

                        with table_col1:

                            with st.container(border=True):

                                st.markdown("### 3️⃣ Full Scaffold Analysis Table")

                                display_paginated_dataframe(
                                    df=scaffold_df,
                                    table_key="enterprise_full_scaffold_analysis_table",
                                    rows_per_page=100
                                )

                        with table_col2:

                            with st.container(border=True):

                                st.markdown("### 4️⃣ Scaffold Frequency Table")

                                display_paginated_dataframe(
                                    df=scaffold_freq_df,
                                    table_key="enterprise_full_scaffold_frequency_table",
                                    rows_per_page=100
                                )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 5️⃣ Top Scaffold Frequency Plot")

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

                            st.pyplot(
                                fig_scaffold
                            )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 6️⃣ Core Structure Explorer")

                            valid_scaffolds = scaffold_freq_df[
                                scaffold_freq_df["Murcko_Scaffold"] != "No Scaffold"
                            ].copy()

                            if valid_scaffolds.empty:

                                st.info(
                                    "No core scaffold structures available to visualize."
                                )

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

                                    core_col1, core_col2 = st.columns([1, 1.4])

                                    with core_col1:

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

                                    with core_col2:

                                        st.markdown("#### Molecules Belonging to Selected Scaffold")

                                        selected_scaffold_molecules = scaffold_df[
                                            scaffold_df["Murcko_Scaffold"] == selected_scaffold
                                        ][
                                            [
                                                "Molecule_Name",
                                                "SMILES",
                                                "Murcko_Scaffold"
                                            ]
                                        ]

                                        display_paginated_dataframe(
                                            df=selected_scaffold_molecules,
                                            table_key="enterprise_selected_scaffold_molecules_table",
                                            rows_per_page=100
                                        )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 7️⃣ Download Scaffold Reports")

                            scaffold_csv = scaffold_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download Processed Scaffold Analysis CSV",
                                data=scaffold_csv,
                                file_name="murcko_scaffold_analysis_processed.csv",
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

            st.error(
                f"Scaffold analysis failed: {e}"
            )


    with tab8:

        st.subheader("🛡️ OOD Detection")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise OOD workflow: select molecule → compare against full chemistry dataset → evaluate similarity, applicability domain, latent distance, prediction reliability, and chemical-space position.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            ood_df = load_molecule_dataset()

            # ==================================================
            # INPUT PANEL
            # ==================================================

            input_col, config_col = st.columns([1.35, 1])

            with input_col:

                with st.container(border=True):

                    st.markdown("### 1️⃣ Select Molecule for OOD Detection")

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

                    if ood_input_mode == "Select from Dataset":

                        st.info(
                            "Search or browse the molecule catalog, then select a molecule for OOD reliability analysis."
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
                                "Reset",
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

                        st.success(
                            f"Matching molecules found: {len(ood_filtered_df)} out of {len(ood_df)}"
                        )

                        display_paginated_molecule_table(
                            df=ood_filtered_df,
                            table_key="enterprise_ood_catalog",
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

                        if ood_filtered_df.empty:

                            st.warning(
                                "No molecule found for this search. Please try another molecule name or SMILES."
                            )

                            st.stop()

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

                    else:

                        ood_query_name = "Custom Input"

                        ood_query_smiles = st.text_input(
                            "Enter SMILES for OOD Detection",
                            value="CCO",
                            key="ood_custom_smiles"
                        )

                    st.markdown("---")

                    st.markdown("### Current OOD Molecule")

                    st.code(
                        f"Molecule: {ood_query_name}\nSMILES: {ood_query_smiles}",
                        language="text"
                    )

                    with st.expander(
                        "Copy OOD molecule details",
                        expanded=False
                    ):

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

            with config_col:

                with st.container(border=True):

                    st.markdown("### 2️⃣ OOD Settings")

                    applicability_threshold = st.slider(
                        "Tanimoto Applicability Domain Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.40,
                        step=0.01,
                        key="enterprise_ood_applicability_threshold"
                    )

                    max_reference_molecules = st.slider(
                        "Reference molecules for latent/PCA/UMAP visual checks",
                        min_value=100,
                        max_value=len(ood_df),
                        value=len(ood_df),
                        step=100,
                        key="enterprise_ood_reference_limit"
                    )

                    st.info(
                        "Similarity is calculated against the full dataset. PCA/UMAP and Mahalanobis can now use the full 3328-molecule reference set; reduce the slider only if cloud performance is slow."
                    )

                    run_ood_detection = st.button(
                        "🚀 Run OOD Detection",
                        key="run_ood_detection",
                        type="primary"
                    )

            if ood_query_smiles is None or str(ood_query_smiles).strip() == "":
                st.error("No valid SMILES available for OOD detection.")
                st.stop()

            # ==================================================
            # RUN OOD DETECTION
            # ==================================================

            if run_ood_detection:

                with st.spinner(
                    "Comparing molecule against known dataset chemistry..."
                ):

                    ood_result = detect_ood_molecule(
                        ood_query_smiles,
                        ood_df
                    )

                    similarity_df = calculate_all_similarity_scores(
                        ood_query_smiles,
                        ood_df
                    )

                max_similarity = ood_result[
                    "Max_Tanimoto_Similarity"
                ]

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 3️⃣ OOD Detection Result Summary")

                    col1, col2, col3, col4 = st.columns(4)

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

                    with col4:

                        if max_similarity is not None:

                            if float(max_similarity) >= applicability_threshold:
                                st.metric(
                                    "Applicability Domain",
                                    "Inside"
                                )
                            else:
                                st.metric(
                                    "Applicability Domain",
                                    "Outside"
                                )

                        else:
                            st.metric(
                                "Applicability Domain",
                                "N/A"
                            )

                    ood_result_df = pd.DataFrame([
                        ood_result
                    ])

                    st.dataframe(
                        ood_result_df,
                        width="stretch"
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

                if similarity_df.empty:

                    st.warning(
                        "Similarity scores could not be calculated."
                    )

                else:

                    top_5_nearest_df = similarity_df.head(5).copy()

                    st.markdown("---")

                    sim_col1, sim_col2 = st.columns([1.2, 1])

                    with sim_col1:

                        with st.container(border=True):

                            st.markdown("### 4️⃣ Top 5 Nearest Molecules")

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

                    with sim_col2:

                        with st.container(border=True):

                            st.markdown("### 5️⃣ Applicability Boundary Gauge")

                            try:

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
                                                "value": applicability_threshold
                                            },
                                            "steps": [
                                                {
                                                    "range": [0, applicability_threshold],
                                                    "color": "lightgray"
                                                },
                                                {
                                                    "range": [applicability_threshold, 1],
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

                                if nearest_similarity_for_gauge >= applicability_threshold:
                                    st.success(
                                        "Molecule is inside the similarity-based applicability domain."
                                    )
                                else:
                                    st.error(
                                        "Molecule is outside the similarity-based applicability domain."
                                    )

                            except Exception as e:

                                st.warning(
                                    f"Applicability boundary visualization failed: {e}"
                                )

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 6️⃣ Similarity Distribution")

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
                        ax_similarity.set_title("Similarity Distribution Against Full Dataset")
                        ax_similarity.legend()

                        st.pyplot(
                            fig_similarity
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

                # ==================================================
                # ADVANCED LATENT DISTANCE
                # ==================================================

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 7️⃣ Mahalanobis Latent Distance")

                    st.write(
                        "PCA-based latent chemical-space distance using Morgan fingerprints. Higher distance means farther from dataset chemical-space center."
                    )

                    mahal_distance, mahal_95, mahal_99 = (
                        calculate_mahalanobis_distance_for_smiles(
                            ood_query_smiles,
                            ood_df,
                            max_reference_molecules=max_reference_molecules,
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

                        latent_status = (
                            "Inside 95% Boundary"
                            if mahal_distance <= mahal_95
                            else (
                                "Between 95% and 99% Boundary"
                                if mahal_distance <= mahal_99
                                else "Outside 99% Boundary"
                            )
                        )

                        if mahal_distance <= mahal_95:
                            st.success(
                                "Mahalanobis result: molecule is inside the 95% latent chemical-space boundary."
                            )
                        elif mahal_distance <= mahal_99:
                            st.warning(
                                "Mahalanobis result: molecule is between the 95% and 99% boundary. Use prediction with caution."
                            )
                        else:
                            st.error(
                                "Mahalanobis result: molecule is outside the 99% latent boundary. This indicates possible OOD chemistry."
                            )

                        mahalanobis_summary_df = pd.DataFrame([{
                            "Query_SMILES": ood_query_smiles,
                            "Mahalanobis_Distance": mahal_distance,
                            "Boundary_95": mahal_95,
                            "Boundary_99": mahal_99,
                            "Latent_Domain_Status": latent_status
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

                # ==================================================
                # ADVANCED PREDICTION RELIABILITY
                # ==================================================

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 8️⃣ Advanced Prediction Reliability inside OOD")

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

                        reliability_col1, reliability_col2 = st.columns(2)

                        with reliability_col1:

                            st.markdown("#### Deep Ensemble Uncertainty")

                            deep_ensemble_ood_df = pd.DataFrame([
                                deep_ensemble_ood
                            ])

                            st.dataframe(
                                deep_ensemble_ood_df,
                                width="stretch"
                            )

                            if deep_ensemble_ood["Uncertainty_Label"] == "Low Uncertainty":
                                st.success(
                                    f"Deep ensemble result: {deep_ensemble_ood['Uncertainty_Label']}"
                                )
                            elif deep_ensemble_ood["Uncertainty_Label"] == "Moderate Uncertainty":
                                st.warning(
                                    f"Deep ensemble result: {deep_ensemble_ood['Uncertainty_Label']}"
                                )
                            else:
                                st.error(
                                    f"Deep ensemble result: {deep_ensemble_ood['Uncertainty_Label']}"
                                )

                        with reliability_col2:

                            st.markdown("#### Conformal Prediction Interval")

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

                            conformal_ood_df = pd.DataFrame([
                                conformal_ood
                            ])

                            st.dataframe(
                                conformal_ood_df,
                                width="stretch"
                            )

                        reliability_summary_df = pd.DataFrame([{
                            "Molecule_Name": ood_query_name,
                            "SMILES": ood_query_smiles,
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
                            label="Download OOD Prediction Reliability Summary CSV",
                            data=reliability_csv,
                            file_name="ood_prediction_reliability_summary.csv",
                            mime="text/csv"
                        )

                    except Exception as e:

                        st.warning(
                            f"Advanced prediction reliability could not be calculated: {e}"
                        )

                # ==================================================
                # NETWORK + CHEMICAL-SPACE VISUALIZATION
                # ==================================================

                st.markdown("---")

                visual_col1, visual_col2 = st.columns(2)

                with visual_col1:

                    with st.container(border=True):

                        st.markdown("### 9️⃣ Interactive Similarity Network Graph")

                        try:

                            if similarity_df.empty:

                                st.warning(
                                    "Similarity network requires similarity scores."
                                )

                            else:

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
                                        molecule_name
                                        if molecule_name != "Name Not Found"
                                        else network_row["SMILES"][:40]
                                    )

                                    graph.add_node(
                                        label,
                                        node_type="Neighbor"
                                    )

                                    graph.add_edge(
                                        query_node,
                                        label,
                                        weight=similarity_value
                                    )

                                pos = nx.spring_layout(
                                    graph,
                                    seed=42
                                )

                                edge_x = []
                                edge_y = []

                                for edge in graph.edges():

                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])

                                edge_trace = go.Scatter(
                                    x=edge_x,
                                    y=edge_y,
                                    line=dict(width=1, color="#888"),
                                    hoverinfo="none",
                                    mode="lines"
                                )

                                node_x = []
                                node_y = []
                                node_text = []
                                node_size = []

                                for node in graph.nodes():

                                    x, y = pos[node]
                                    node_x.append(x)
                                    node_y.append(y)
                                    node_text.append(node)
                                    node_size.append(
                                        28 if node == query_node else 18
                                    )

                                node_trace = go.Scatter(
                                    x=node_x,
                                    y=node_y,
                                    mode="markers+text",
                                    text=node_text,
                                    textposition="top center",
                                    hoverinfo="text",
                                    marker=dict(
                                        showscale=False,
                                        size=node_size,
                                        line_width=2
                                    )
                                )

                                network_fig = go.Figure(
                                    data=[
                                        edge_trace,
                                        node_trace
                                    ]
                                )

                                network_fig.update_layout(
                                    title="Query Molecule Similarity Network",
                                    showlegend=False,
                                    height=550,
                                    margin=dict(l=20, r=20, t=50, b=20)
                                )

                                st.plotly_chart(
                                    network_fig,
                                    width="stretch"
                                )

                        except Exception as e:

                            st.warning(
                                f"Similarity network graph failed: {e}"
                            )

                with visual_col2:

                    with st.container(border=True):

                        st.markdown("### 🔟 PCA Chemical-Space Visualization inside OOD")

                        try:

                            pca_ood_df = generate_ood_chemical_space_embeddings(
                                query_smiles=ood_query_smiles,
                                molecule_df=ood_df,
                                max_reference_molecules=max_reference_molecules,
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

                with st.container(border=True):

                    st.markdown("### 1️⃣1️⃣ UMAP Nearest-Neighbor Visualization inside OOD")

                    if not UMAP_AVAILABLE:

                        st.warning(
                            "UMAP is not installed. Please install it first: pip install umap-learn"
                        )

                    else:

                        try:

                            umap_ood_df = generate_ood_chemical_space_embeddings(
                                query_smiles=ood_query_smiles,
                                molecule_df=ood_df,
                                max_reference_molecules=max_reference_molecules,
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
                                    title="OOD UMAP Nearest-Neighbor Visualization"
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

        except Exception as e:

            st.error(
                f"OOD detection failed: {e}"
            )


    with tab9:

        st.subheader("📉 PCA Chemical Space")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise PCA workflow: select molecule → configure chemical-space sample → generate interactive PCA plot → inspect outliers and download coordinates.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            pca_full_df = load_molecule_dataset()

            # ==================================================
            # INPUT + SETTINGS
            # ==================================================

            input_col, settings_col = st.columns([1.35, 1])

            with input_col:

                with st.container(border=True):

                    st.markdown("### 1️⃣ PCA Input Mode")

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
                                "Reset",
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

                        st.success(
                            f"Matching molecules found: {len(pca_filtered_df)} out of {len(pca_full_df)}"
                        )

                        display_paginated_molecule_table(
                            df=pca_filtered_df,
                            table_key="enterprise_pca_catalog",
                            rows_per_page=100,
                            columns=[
                                "Molecule_Name",
                                "SMILES"
                            ]
                        )

                        if pca_filtered_df.empty:

                            st.warning(
                                "No molecule found. Please try another search."
                            )

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

                            st.markdown("### Selected Molecule")

                            st.code(
                                f"Molecule: {selected_pca_name}\nSMILES: {selected_pca_smiles}",
                                language="text"
                            )

            with settings_col:

                with st.container(border=True):

                    st.markdown("### 2️⃣ PCA Settings")

                    sample_size_pca = st.slider(
                        "Number of molecules to visualize",
                        min_value=100,
                        max_value=len(pca_full_df),
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

                    st.info(
                        "PCA supports full dataset visualization. On Streamlit Cloud, start with 1000 molecules and increase gradually."
                    )

                    run_pca = st.button(
                        "🚀 Generate Interactive PCA Chemical Space Plot",
                        key="generate_interactive_pca_plot",
                        type="primary"
                    )

            # ==================================================
            # RUN PCA
            # ==================================================

            if run_pca:

                with st.spinner(
                    "Generating interactive PCA chemical space..."
                ):

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

                    st.success(
                        "Interactive PCA chemical space generated successfully."
                    )

                    pc1_var = pca_df[
                        "Explained_Variance_PC1_%"
                    ].iloc[0]

                    pc2_var = pca_df[
                        "Explained_Variance_PC2_%"
                    ].iloc[0]

                    outlier_count = int(
                        (
                            pca_df["Outlier_Status"] == "Potential Outlier"
                        ).sum()
                    )

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 3️⃣ PCA Summary Metrics")

                        col_pca1, col_pca2, col_pca3, col_pca4 = st.columns(4)

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

                        with col_pca4:
                            st.metric(
                                "Potential Outliers",
                                outlier_count
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

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 4️⃣ Interactive PCA Chemical Space Plot")

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

                    pca_detail_col1, pca_detail_col2 = st.columns(2)

                    with pca_detail_col1:

                        with st.container(border=True):

                            st.markdown("### 5️⃣ Selected Molecule PCA Location")

                            selected_rows = pca_df[
                                pca_df["Point_Type"] == "Selected Molecule"
                            ]

                            if selected_rows.empty:

                                st.info(
                                    "No single molecule selected. Use Single Molecule PCA Search mode to highlight one molecule."
                                )

                            else:

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

                    with pca_detail_col2:

                        with st.container(border=True):

                            st.markdown("### 6️⃣ PCA Outlier Detection Table")

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

                                st.info(
                                    "No PCA outliers detected."
                                )

                            else:

                                st.warning(
                                    f"{len(outlier_df)} potential PCA outlier(s) detected."
                                )

                                display_paginated_dataframe(
                                    df=outlier_df,
                                    table_key="enterprise_pca_outlier_table",
                                    rows_per_page=100
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

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 7️⃣ Download PCA Coordinates")

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

                        pca_csv = pca_df[
                            download_cols
                        ].to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download PCA Coordinates CSV",
                            data=pca_csv,
                            file_name="interactive_pca_chemical_space_coordinates.csv",
                            mime="text/csv"
                        )

                        st.info(
                            "Interpretation: PCA shows global molecular distribution. Points far from dense regions or flagged as potential outliers may represent unusual chemistry compared with the dataset."
                        )

        except Exception as e:

            st.error(
                f"PCA chemical space visualization failed: {e}"
            )


    with tab10:

        st.subheader("🌀 t-SNE Chemical Space")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise t-SNE workflow: select molecule → configure nonlinear neighborhood map → visualize chemical clusters → inspect isolated/outlier molecules.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            tsne_full_df = load_molecule_dataset()

            # ==================================================
            # INPUT + SETTINGS
            # ==================================================

            input_col, settings_col = st.columns([1.35, 1])

            with input_col:

                with st.container(border=True):

                    st.markdown("### 1️⃣ t-SNE Input Mode")

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
                                "Reset",
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

                        st.success(
                            f"Matching molecules found: {len(tsne_filtered_df)} out of {len(tsne_full_df)}"
                        )

                        display_paginated_molecule_table(
                            df=tsne_filtered_df,
                            table_key="enterprise_tsne_catalog",
                            rows_per_page=100,
                            columns=[
                                "Molecule_Name",
                                "SMILES"
                            ]
                        )

                        if tsne_filtered_df.empty:

                            st.warning(
                                "No molecule found. Please try another search."
                            )

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

                            st.markdown("### Selected Molecule")

                            st.code(
                                f"Molecule: {selected_tsne_name}\nSMILES: {selected_tsne_smiles}",
                                language="text"
                            )

            with settings_col:

                with st.container(border=True):

                    st.markdown("### 2️⃣ t-SNE Settings")

                    sample_size_tsne = st.slider(
                        "Number of molecules to visualize",
                        min_value=100,
                        max_value=len(tsne_full_df),
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

                    st.warning(
                        "Cloud safety note: t-SNE is computationally heavier than PCA. Start with 500–1000 molecules, then increase gradually."
                    )

                    run_tsne = st.button(
                        "🚀 Generate Interactive t-SNE Chemical Space Plot",
                        key="generate_interactive_tsne_plot",
                        type="primary"
                    )

            # ==================================================
            # RUN t-SNE
            # ==================================================

            if run_tsne:

                with st.spinner(
                    "Generating interactive t-SNE chemical space..."
                ):

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

                    st.success(
                        "Interactive t-SNE chemical space generated successfully."
                    )

                    outlier_count = int(
                        (
                            tsne_df["TSNE_Outlier_Status"]
                            ==
                            "Potential Outlier"
                        ).sum()
                    )

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 3️⃣ t-SNE Summary Metrics")

                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)

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
                                "Iterations",
                                tsne_iterations
                            )

                        with col_t4:
                            st.metric(
                                "Potential Outliers",
                                outlier_count
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

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 4️⃣ Interactive t-SNE Chemical Space Plot")

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

                    tsne_detail_col1, tsne_detail_col2 = st.columns(2)

                    with tsne_detail_col1:

                        with st.container(border=True):

                            st.markdown("### 5️⃣ Selected Molecule t-SNE Location")

                            selected_rows = tsne_df[
                                tsne_df["Point_Type"] == "Selected Molecule"
                            ]

                            if selected_rows.empty:

                                st.info(
                                    "No single molecule selected. Use Single Molecule t-SNE Search mode to highlight one molecule."
                                )

                            else:

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

                    with tsne_detail_col2:

                        with st.container(border=True):

                            st.markdown("### 6️⃣ t-SNE Potential Outlier Table")

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

                                st.info(
                                    "No t-SNE outliers detected."
                                )

                            else:

                                st.warning(
                                    f"{len(outlier_df)} potential t-SNE outlier(s) detected."
                                )

                                display_paginated_dataframe(
                                    df=outlier_df,
                                    table_key="enterprise_tsne_outlier_table",
                                    rows_per_page=100
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

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 7️⃣ Download t-SNE Coordinates")

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

                        tsne_csv = tsne_df[
                            download_cols
                        ].to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download t-SNE Coordinates CSV",
                            data=tsne_csv,
                            file_name="interactive_tsne_chemical_space_coordinates.csv",
                            mime="text/csv"
                        )

                        st.info(
                            "Interpretation: t-SNE is best for local molecular neighborhood patterns. Close points are likely fingerprint-similar, while isolated points may represent unusual chemistry."
                        )

        except Exception as e:

            st.error(
                f"t-SNE chemical space visualization failed: {e}"
            )


    with tab11:

        st.subheader("🌌 UMAP Chemical Space")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise UMAP workflow: select molecule → configure nonlinear manifold map → visualize chemical neighborhoods → inspect cluster/outlier behavior.
            </div>
            """,
            unsafe_allow_html=True
        )

        if not UMAP_AVAILABLE:

            with st.container(border=True):

                st.markdown("### UMAP Dependency Missing")

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

                # ==================================================
                # INPUT + SETTINGS
                # ==================================================

                input_col, settings_col = st.columns([1.35, 1])

                with input_col:

                    with st.container(border=True):

                        st.markdown("### 1️⃣ UMAP Input Mode")

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
                                    "Reset",
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

                            st.success(
                                f"Matching molecules found: {len(umap_filtered_df)} out of {len(umap_full_df)}"
                            )

                            display_paginated_molecule_table(
                                df=umap_filtered_df,
                                table_key="enterprise_umap_catalog",
                                rows_per_page=100,
                                columns=[
                                    "Molecule_Name",
                                    "SMILES"
                                ]
                            )

                            if umap_filtered_df.empty:

                                st.warning(
                                    "No molecule found. Please try another search."
                                )

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

                                st.markdown("### Selected Molecule")

                                st.code(
                                    f"Molecule: {selected_umap_name}\nSMILES: {selected_umap_smiles}",
                                    language="text"
                                )

                with settings_col:

                    with st.container(border=True):

                        st.markdown("### 2️⃣ UMAP Settings")

                        sample_size_umap = st.slider(
                            "Number of molecules to visualize",
                            min_value=100,
                            max_value=len(umap_full_df),
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
                            "UMAP supports full dataset visualization. On Streamlit Cloud, start with 1000–2000 molecules and increase gradually."
                        )

                        run_umap = st.button(
                            "🚀 Generate Interactive UMAP Chemical Space Plot",
                            key="generate_interactive_umap_plot",
                            type="primary"
                        )

                # ==================================================
                # RUN UMAP
                # ==================================================

                if run_umap:

                    with st.spinner(
                        "Generating interactive UMAP chemical space..."
                    ):

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

                        st.success(
                            "Interactive UMAP chemical space generated successfully."
                        )

                        outlier_count = int(
                            (
                                umap_df["UMAP_Outlier_Status"]
                                ==
                                "Potential Outlier"
                            ).sum()
                        )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 3️⃣ UMAP Summary Metrics")

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
                                    outlier_count
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

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 4️⃣ Interactive UMAP Chemical Space Plot")

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

                        umap_detail_col1, umap_detail_col2 = st.columns(2)

                        with umap_detail_col1:

                            with st.container(border=True):

                                st.markdown("### 5️⃣ Selected Molecule UMAP Location")

                                selected_rows = umap_df[
                                    umap_df["Point_Type"] == "Selected Molecule"
                                ]

                                if selected_rows.empty:

                                    st.info(
                                        "No single molecule selected. Use Single Molecule UMAP Search mode to highlight one molecule."
                                    )

                                else:

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

                        with umap_detail_col2:

                            with st.container(border=True):

                                st.markdown("### 6️⃣ UMAP Potential Outlier Table")

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

                                    st.info(
                                        "No UMAP outliers detected."
                                    )

                                else:

                                    st.warning(
                                        f"{len(outlier_df)} potential UMAP outlier(s) detected."
                                    )

                                    display_paginated_dataframe(
                                        df=outlier_df,
                                        table_key="enterprise_umap_outlier_table",
                                        rows_per_page=100
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

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 7️⃣ Download UMAP Coordinates")

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

                            umap_csv = umap_df[
                                download_cols
                            ].to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download UMAP Coordinates CSV",
                                data=umap_csv,
                                file_name="interactive_umap_chemical_space_coordinates.csv",
                                mime="text/csv"
                            )

                            st.info(
                                "Interpretation: UMAP is useful for nonlinear chemical-space neighborhoods and scaffold-like clustering. Isolated regions may indicate unusual chemistry or potential OOD behavior."
                            )

            except Exception as e:

                st.error(
                    f"UMAP chemical space visualization failed: {e}"
                )


    with tab12:

        st.subheader("⚡ Interactive Plotly UMAP + AI Overlay")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise interactive UMAP workflow: select molecule → configure UMAP → optionally add AI prediction/OOD overlay → explore clusters, outliers, and downloadable interactive HTML.
            </div>
            """,
            unsafe_allow_html=True
        )

        if not UMAP_AVAILABLE:

            with st.container(border=True):

                st.markdown("### UMAP Dependency Missing")

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

                # ==================================================
                # INPUT + SETTINGS
                # ==================================================

                input_col, settings_col = st.columns([1.35, 1])

                with input_col:

                    with st.container(border=True):

                        st.markdown("### 1️⃣ Interactive UMAP Input Mode")

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

                        if plotly_umap_mode == "Single Molecule Interactive UMAP Search":

                            st.info(
                                "Search/select one molecule. The selected molecule will be highlighted as a star on the interactive Plotly UMAP plot."
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
                                    "Reset",
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

                            st.success(
                                f"Matching molecules found: {len(plotly_umap_filtered_df)} out of {len(plotly_umap_full_df)}"
                            )

                            display_paginated_molecule_table(
                                df=plotly_umap_filtered_df,
                                table_key="enterprise_plotly_umap_catalog",
                                rows_per_page=100,
                                columns=[
                                    "Molecule_Name",
                                    "SMILES"
                                ]
                            )

                            if plotly_umap_filtered_df.empty:

                                st.warning(
                                    "No molecule found. Please try another search."
                                )

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

                                st.markdown("### Selected Molecule")

                                st.code(
                                    f"Molecule: {selected_plotly_umap_name}\nSMILES: {selected_plotly_umap_smiles}",
                                    language="text"
                                )

                with settings_col:

                    with st.container(border=True):

                        st.markdown("### 2️⃣ Plotly UMAP + AI Overlay Settings")

                        sample_size_plotly_umap = st.slider(
                            "Number of molecules for interactive Plotly UMAP",
                            min_value=100,
                            max_value=len(plotly_umap_full_df),
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
                                "AI overlay mode calculates predictions/OOD for every plotted molecule. Use 100–500 molecules first for speed."
                            )

                        else:

                            st.info(
                                "Standard Plotly UMAP can use larger sample sizes. On Streamlit Cloud, start with 1000–2000 molecules and increase gradually."
                            )

                        run_plotly_umap = st.button(
                            "🚀 Generate Interactive Plotly UMAP",
                            key="generate_updated_interactive_plotly_umap",
                            type="primary"
                        )

                # ==================================================
                # RUN INTERACTIVE PLOTLY UMAP
                # ==================================================

                if run_plotly_umap:

                    with st.spinner(
                        "Generating updated interactive Plotly UMAP..."
                    ):

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

                            overlay_progress = st.progress(0)
                            overlay_status = st.empty()

                            for i, (_, overlay_row) in enumerate(plotly_umap_df.iterrows()):

                                overlay_smiles = overlay_row["SMILES"]

                                overlay_status.info(
                                    f"Calculating AI overlay {i + 1} of {len(plotly_umap_df)}"
                                )

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

                                overlay_progress.progress(
                                    int((i + 1) / len(plotly_umap_df) * 100)
                                )

                            overlay_status.empty()

                            ai_overlay_df = pd.DataFrame(
                                ai_overlay_rows
                            )

                            plotly_umap_df = plotly_umap_df.merge(
                                ai_overlay_df,
                                on="SMILES",
                                how="left"
                            )

                    if plotly_umap_df.empty:

                        st.warning(
                            "Interactive Plotly UMAP could not be generated. Not enough valid molecules."
                        )

                    else:

                        st.success(
                            "Updated Interactive Plotly UMAP generated successfully."
                        )

                        outlier_count = int(
                            (
                                plotly_umap_df["UMAP_Outlier_Status"]
                                ==
                                "Potential Outlier"
                            ).sum()
                        )

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 3️⃣ Interactive UMAP Summary Metrics")

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
                                    outlier_count
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

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 4️⃣ Interactive Plotly UMAP Explorer")

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

                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:

                            with st.container(border=True):

                                st.markdown("### 5️⃣ Selected Molecule UMAP Location")

                                selected_rows = plotly_umap_df[
                                    plotly_umap_df["Point_Type"] == "Selected Molecule"
                                ]

                                if selected_rows.empty:

                                    st.info(
                                        "No single molecule selected. Use Single Molecule Interactive UMAP Search mode to highlight one molecule."
                                    )

                                else:

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

                        with detail_col2:

                            with st.container(border=True):

                                st.markdown("### 6️⃣ UMAP Potential Outlier Table")

                                outlier_df = plotly_umap_df[
                                    plotly_umap_df["UMAP_Outlier_Status"] == "Potential Outlier"
                                ].sort_values(
                                    by="UMAP_Distance_From_Center",
                                    ascending=False
                                )

                                if outlier_df.empty:

                                    st.info(
                                        "No UMAP outliers detected."
                                    )

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

                                    display_paginated_dataframe(
                                        df=outlier_df[outlier_display_cols],
                                        table_key="enterprise_plotly_umap_outliers",
                                        rows_per_page=100
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

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 7️⃣ Download Interactive UMAP Data")

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
                                "Interpretation: Interactive UMAP supports zooming, panning, hovering, selected molecule highlighting, scaffold exploration, outlier review, and optional AI overlay coloring."
                            )

            except Exception as e:

                st.error(
                    f"Interactive Plotly UMAP visualization failed: {e}"
                )


    with tab13:

        st.subheader("💊 Drug-Likeness Analysis")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise drug-discovery workflow: select molecule or batch dataset → evaluate Lipinski, Veber, Ghose, lead-likeness, PAINS, synthetic accessibility, radar profile, and downloadable reports.
            </div>
            """,
            unsafe_allow_html=True
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

                input_col, summary_col = st.columns([1.35, 1])

                with input_col:

                    with st.container(border=True):

                        st.markdown("### 1️⃣ Select Molecule for Drug-Likeness")

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
                                "Search or browse the molecule catalog, then select a molecule for drug-likeness screening."
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

                            col_drug_search, col_drug_reset = st.columns([5, 1])

                            with col_drug_search:

                                drug_search_query = st.text_input(
                                    "Search molecule by IUPAC/name or SMILES",
                                    value="",
                                    key=drug_search_key,
                                    placeholder="Example: aspirin, ethanol, benz, acid, CCO"
                                )

                            with col_drug_reset:

                                st.write("")
                                st.write("")

                                if st.button(
                                    "Reset",
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

                            st.success(
                                f"Matching molecules found: {len(drug_filtered_df)} out of {len(drug_df)}"
                            )

                            display_paginated_molecule_table(
                                df=drug_filtered_df,
                                table_key="enterprise_drug_catalog",
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

                            if drug_filtered_df.empty:

                                st.warning(
                                    "No molecule found for this search. Please try another molecule name or SMILES."
                                )

                                st.stop()

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

                            st.session_state["drug_selected_molecule_name"] = drug_name
                            st.session_state["drug_selected_smiles"] = drug_smiles

                        else:

                            drug_name = "Custom Input"

                            drug_smiles = st.text_input(
                                "Enter custom SMILES",
                                value="CCO",
                                key="drug_likeness_custom_smiles"
                            )

                        st.markdown("---")

                        st.markdown("### Selected Molecule")

                        st.code(
                            f"Molecule: {drug_name}\nSMILES: {drug_smiles}",
                            language="text"
                        )

                        with st.expander(
                            "Copy selected molecule details",
                            expanded=False
                        ):

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

                        run_drug_single = st.button(
                            "🚀 Run Single Molecule Drug-Likeness Analysis",
                            key="run_single_drug_likeness_analysis",
                            type="primary"
                        )

                with summary_col:

                    with st.container(border=True):

                        st.markdown("### 2️⃣ Screening Rules Included")

                        rules_overview_df = pd.DataFrame({
                            "Module": [
                                "Lipinski Rule of 5",
                                "Veber Rule",
                                "Ghose Filter",
                                "Lead-Likeness",
                                "PAINS Alerts",
                                "Synthetic Accessibility",
                                "Radar Chart"
                            ],
                            "Purpose": [
                                "Core oral drug-likeness",
                                "Permeability/flexibility",
                                "Drug-like property window",
                                "Early lead filtering",
                                "Problematic assay motifs",
                                "Ease-of-synthesis estimate",
                                "Visual property balance"
                            ]
                        })

                        st.dataframe(
                            rules_overview_df,
                            width="stretch"
                        )

                        st.info(
                            "Drug-likeness is a screening guide, not a guarantee of biological activity or safety."
                        )

                if run_drug_single:

                    mol = Chem.MolFromSmiles(
                        drug_smiles
                    )

                    if mol is None:

                        st.error(
                            "Invalid SMILES. Please enter or select a valid molecule."
                        )

                        st.stop()

                    with st.spinner(
                        "Calculating drug-likeness descriptors and screening rules..."
                    ):

                        drug_properties = calculate_drug_likeness_properties(
                            drug_smiles
                        )

                    if drug_properties is None:

                        st.error(
                            "Drug-likeness calculation failed for this molecule."
                        )

                        st.stop()

                    report_df = pd.DataFrame([
                        drug_properties
                    ])

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 3️⃣ Drug-Likeness KPI Summary")

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
                                "PAINS",
                                drug_properties["PAINS_Status"]
                            )

                        with col_d4:
                            st.metric(
                                "Drug-Likeness",
                                drug_properties["Drug_Likeness_Label"]
                            )

                        col_d5, col_d6, col_d7, col_d8 = st.columns(4)

                        with col_d5:
                            st.metric(
                                "Molecular Weight",
                                drug_properties["Molecular_Weight"]
                            )

                        with col_d6:
                            st.metric(
                                "LogP",
                                drug_properties["LogP"]
                            )

                        with col_d7:
                            st.metric(
                                "TPSA",
                                drug_properties["TPSA"]
                            )

                        with col_d8:
                            st.metric(
                                "Rotatable Bonds",
                                drug_properties["Rotatable_Bonds"]
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
                            st.success(
                                "PAINS screening: no PAINS alerts detected."
                            )
                        elif drug_properties["PAINS_Status"] == "PAINS Screening Unavailable":
                            st.warning(
                                "PAINS screening unavailable in current RDKit environment."
                            )
                        else:
                            st.error(
                                f"PAINS screening: {drug_properties['PAINS_Alert_Count']} alert(s) detected."
                            )

                        st.info(
                            drug_properties["Bioavailability_Label"]
                        )

                    st.markdown("---")

                    chart_col, descriptor_col = st.columns([1, 1.2])

                    with chart_col:

                        with st.container(border=True):

                            st.markdown("### 4️⃣ Drug-Likeness Radar Chart")

                            radar_metrics = {
                                "MW": max(0, min(100, 100 - abs(drug_properties["Molecular_Weight"] - 350) / 350 * 100)),
                                "LogP": max(0, min(100, 100 - abs(drug_properties["LogP"] - 2.5) / 5 * 100)),
                                "TPSA": max(0, min(100, 100 - drug_properties["TPSA"] / 140 * 100)),
                                "RotB": max(0, min(100, 100 - drug_properties["Rotatable_Bonds"] / 10 * 100)),
                                "HBD": max(0, min(100, 100 - drug_properties["H_Bond_Donors"] / 5 * 100)),
                                "HBA": max(0, min(100, 100 - drug_properties["H_Bond_Acceptors"] / 10 * 100))
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

                    with descriptor_col:

                        with st.container(border=True):

                            st.markdown("### 5️⃣ Molecular Descriptor Table")

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

                    detail_col1, detail_col2 = st.columns([1.2, 1])

                    with detail_col1:

                        with st.container(border=True):

                            st.markdown("### 6️⃣ Rule-Based Drug-Likeness Details")

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

                    with detail_col2:

                        with st.container(border=True):

                            st.markdown("### 7️⃣ Interpretation")

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

                    st.markdown("---")

                    with st.container(border=True):

                        st.markdown("### 8️⃣ Download Reports")

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

                with st.container(border=True):

                    st.markdown("### 1️⃣ Batch Drug-Likeness Input")

                    st.info(
                        "Analyze uploaded CSV molecules or use the full 3328-molecule dataset. For Streamlit Cloud, start with 500 molecules and increase gradually."
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

                        st.success(
                            f"Using full dataset: {len(batch_input_df)} molecules."
                        )

                    elif uploaded_drug_csv is not None:

                        batch_input_df = pd.read_csv(
                            uploaded_drug_csv
                        )

                        st.success(
                            f"Uploaded CSV loaded: {batch_input_df.shape[0]} rows."
                        )

                    if batch_input_df is not None:

                        display_paginated_dataframe(
                            df=batch_input_df,
                            table_key="enterprise_batch_drug_input_preview",
                            rows_per_page=100
                        )

                if batch_input_df is not None:

                    if "SMILES" not in batch_input_df.columns:

                        st.error(
                            "CSV must contain a `SMILES` column."
                        )

                    else:

                        st.markdown("---")

                        with st.container(border=True):

                            st.markdown("### 2️⃣ Batch Analysis Settings")

                            max_batch_rows = st.slider(
                                "Maximum molecules to analyze",
                                min_value=10,
                                max_value=len(batch_input_df),
                                value=min(500, len(batch_input_df)),
                                step=50,
                                key="batch_drug_likeness_max_rows"
                            )

                            st.warning(
                                "Cloud safety note: start with 500 molecules. Increase gradually if processing remains stable."
                            )

                            run_batch_drug = st.button(
                                "🚀 Run Batch Drug-Likeness Analysis",
                                key="run_batch_drug_likeness_analysis",
                                type="primary"
                            )

                        if run_batch_drug:

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

                                st.markdown("---")

                                with st.container(border=True):

                                    st.markdown("### 3️⃣ Batch Drug-Likeness Summary")

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

                                    st.success(
                                        f"Batch drug-likeness analysis completed for {len(batch_result_df)} molecules."
                                    )

                                st.markdown("---")

                                with st.container(border=True):

                                    st.markdown("### 4️⃣ Batch Drug-Likeness Results")

                                    display_paginated_dataframe(
                                        df=batch_result_df,
                                        table_key="enterprise_batch_drug_results",
                                        rows_per_page=100
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

            st.error(
                f"Drug-likeness analysis failed: {e}"
            )



    # ==================================================
    # TAB 14 — EXPLAINABLE AI
    # ==================================================

    with tab14:

        st.subheader("🧠 Explainable AI")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Enterprise Explainable AI workflow: select molecule → generate prediction → explain descriptor contributions → compare model drivers → export interpretation reports.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            xai_df = load_molecule_dataset()

            # ==================================================
            # INPUT + MODULE OVERVIEW
            # ==================================================

            xai_input_col, xai_overview_col = st.columns([1.35, 1])

            with xai_input_col:

                with st.container(border=True):

                    st.markdown("### 1️⃣ Select Molecule for Explanation")

                    xai_input_mode = st.radio(
                        "Choose Explainability Input Method",
                        [
                            "Select from Dataset",
                            "Enter Custom SMILES"
                        ],
                        horizontal=True,
                        key="xai_input_mode"
                    )

                    selected_xai_name = "Custom Input"
                    selected_xai_smiles = "CCO"

                    if xai_input_mode == "Select from Dataset":

                        st.info(
                            "Search or browse the molecule catalog, then select a molecule for model explanation."
                        )

                        if "xai_search_reset_counter" not in st.session_state:
                            st.session_state["xai_search_reset_counter"] = 0

                        xai_search_key = (
                            "xai_search_"
                            + str(st.session_state["xai_search_reset_counter"])
                        )

                        xai_select_key = (
                            "xai_select_"
                            + str(st.session_state["xai_search_reset_counter"])
                        )

                        col_xai_search, col_xai_reset = st.columns([5, 1])

                        with col_xai_search:

                            xai_search_query = st.text_input(
                                "Search molecule by IUPAC/name or SMILES",
                                value="",
                                key=xai_search_key,
                                placeholder="Example: ethanol, benzene, aspirin, CCO"
                            )

                        with col_xai_reset:

                            st.write("")
                            st.write("")

                            if st.button(
                                "Reset",
                                key="reset_xai_search"
                            ):

                                for key_to_clear in [
                                    "xai_selected_molecule_name",
                                    "xai_selected_smiles"
                                ]:
                                    if key_to_clear in st.session_state:
                                        del st.session_state[key_to_clear]

                                st.session_state["xai_search_reset_counter"] += 1
                                st.rerun()

                        if xai_search_query.strip() != "":

                            xai_filtered_df = xai_df[
                                xai_df["Molecule_Name"].str.contains(
                                    xai_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                                |
                                xai_df["SMILES"].str.contains(
                                    xai_search_query,
                                    case=False,
                                    na=False,
                                    regex=False
                                )
                            ].copy()

                        else:

                            xai_filtered_df = xai_df.copy()

                        st.success(
                            f"Matching molecules found: {len(xai_filtered_df)} out of {len(xai_df)}"
                        )

                        display_paginated_molecule_table(
                            df=xai_filtered_df,
                            table_key="enterprise_xai_catalog",
                            rows_per_page=100,
                            columns=[
                                "Molecule_Name",
                                "SMILES"
                            ]
                        )

                        if xai_filtered_df.empty:

                            st.warning(
                                "No molecule found. Please try another search."
                            )

                            st.stop()

                        selected_xai_index = st.selectbox(
                            "Select molecule for explanation",
                            options=xai_filtered_df.index,
                            format_func=lambda x: xai_filtered_df.loc[
                                x,
                                "Molecule_Display"
                            ],
                            key=xai_select_key
                        )

                        selected_xai_name = xai_filtered_df.loc[
                            selected_xai_index,
                            "Molecule_Name"
                        ]

                        selected_xai_smiles = xai_filtered_df.loc[
                            selected_xai_index,
                            "SMILES"
                        ]

                        st.session_state["xai_selected_molecule_name"] = (
                            selected_xai_name
                        )

                        st.session_state["xai_selected_smiles"] = (
                            selected_xai_smiles
                        )

                    else:

                        selected_xai_name = st.text_input(
                            "Molecule name / label",
                            value="Custom Input",
                            key="xai_custom_name"
                        )

                        selected_xai_smiles = st.text_input(
                            "Enter custom SMILES",
                            value="CCO",
                            key="xai_custom_smiles"
                        )

                    st.markdown("---")

                    st.markdown("### Selected Molecule")

                    st.code(
                        f"Molecule: {selected_xai_name}\nSMILES: {selected_xai_smiles}",
                        language="text"
                    )

                    xai_model_choice = st.radio(
                        "Choose Explanation Target",
                        [
                            "RDKit LightGBM Explanation",
                            "Hybrid GAT Explanation",
                            "Ensemble Summary Explanation"
                        ],
                        key="xai_model_choice"
                    )

                    run_xai = st.button(
                        "🚀 Run Explainable AI Analysis",
                        type="primary",
                        key="run_xai_analysis"
                    )

            with xai_overview_col:

                with st.container(border=True):

                    st.markdown("### 2️⃣ Explainability Modules")

                    explainability_df = pd.DataFrame({
                        "Module": [
                            "Prediction Summary",
                            "RDKit SHAP Explanation",
                            "Hybrid GAT Explanation",
                            "Feature Importance",
                            "Positive Drivers",
                            "Negative Drivers",
                            "Scientific Interpretation",
                            "CSV Export"
                        ],
                        "Purpose": [
                            "Model output",
                            "Descriptor-level contribution",
                            "Hybrid model explanation",
                            "Global/local drivers",
                            "Increase prediction",
                            "Decrease prediction",
                            "Human-readable explanation",
                            "Downloadable audit report"
                        ]
                    })

                    st.dataframe(
                        explainability_df,
                        width="stretch"
                    )

                    st.info(
                        "This tab uses your existing explanation functions, so it does not disturb model loading, 3D visualization, or prediction backend code."
                    )

            # ==================================================
            # RUN XAI
            # ==================================================

            if run_xai:

                mol = Chem.MolFromSmiles(
                    selected_xai_smiles
                )

                if mol is None:

                    st.error(
                        "Invalid SMILES. Please enter or select a valid molecule."
                    )

                    st.stop()

                with st.spinner(
                    "Generating prediction and explainability outputs..."
                ):

                    rdkit_prediction = float(
                        predict_melting_point(selected_xai_smiles)
                    )

                    hybrid_prediction = float(
                        predict_hybrid_gat(selected_xai_smiles)
                    )

                    ensemble_prediction = (
                        0.4 * rdkit_prediction
                        +
                        0.6 * hybrid_prediction
                    )

                    uncertainty_xai = calculate_prediction_uncertainty(
                        rdkit_prediction,
                        hybrid_prediction
                    )

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 3️⃣ Prediction Summary")

                    col_x1, col_x2, col_x3, col_x4 = st.columns(4)

                    with col_x1:
                        st.metric(
                            "RDKit LightGBM",
                            f"{rdkit_prediction:.2f} K"
                        )

                    with col_x2:
                        st.metric(
                            "Hybrid GAT",
                            f"{hybrid_prediction:.2f} K"
                        )

                    with col_x3:
                        st.metric(
                            "Ensemble",
                            f"{ensemble_prediction:.2f} K"
                        )

                    with col_x4:
                        st.metric(
                            "Confidence",
                            f"{uncertainty_xai['confidence']}%"
                        )

                    st.info(
                        f"Model difference: {uncertainty_xai['difference']:.2f} K | "
                        f"Estimated uncertainty: ±{uncertainty_xai['uncertainty_range']:.2f} K | "
                        f"{uncertainty_xai['confidence_label']}"
                    )

                # ==================================================
                # STRUCTURE + BASIC DESCRIPTORS
                # ==================================================

                st.markdown("---")

                structure_col, descriptor_col = st.columns([1.35, 1.2])

                with structure_col:

                    with st.container(border=True):

                        st.markdown("### 4️⃣ Molecule Structure")

                        xai_2d_col, xai_3d_col = st.columns(2)

                        with xai_2d_col:

                            st.markdown("#### 2D Structure")

                            molecule_image = Draw.MolToImage(
                                mol,
                                size=(360, 360)
                            )

                            st.image(
                                molecule_image,
                                caption=f"2D Structure: {selected_xai_name}"
                            )

                        with xai_3d_col:

                            st.markdown("#### 3D Structure")

                            st.info(
                                "3D structure uses the existing py3Dmol viewer function. No original 3D code was changed."
                            )

                            show_3d_molecule(
                                selected_xai_smiles,
                                width=430,
                                height=360,
                                viewer_key=f"xai_3d_{make_safe_filename(selected_xai_name)}"
                            )

                with descriptor_col:

                    with st.container(border=True):

                        st.markdown("### 5️⃣ Descriptor Snapshot")

                        descriptor_snapshot_df = pd.DataFrame({
                            "Descriptor": [
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

                        descriptor_snapshot_df["Value"] = descriptor_snapshot_df[
                            "Value"
                        ].astype(str)

                        st.dataframe(
                            descriptor_snapshot_df,
                            width="stretch"
                        )

                # ==================================================
                # EXPLANATION OUTPUTS
                # ==================================================

                st.markdown("---")

                explanation_col, chart_col = st.columns([1.1, 1])

                explanation_df = pd.DataFrame()
                hybrid_importance_df = pd.DataFrame()

                with explanation_col:

                    with st.container(border=True):

                        st.markdown("### 6️⃣ Local Explanation Table")

                        try:

                            if xai_model_choice == "RDKit LightGBM Explanation":

                                explanation_df = explain_prediction(
                                    selected_xai_smiles
                                )

                                st.dataframe(
                                    explanation_df,
                                    width="stretch"
                                )

                            elif xai_model_choice == "Hybrid GAT Explanation":

                                explanation_df = explain_hybrid_gat_prediction(
                                    selected_xai_smiles,
                                    top_n=15
                                )

                                st.dataframe(
                                    explanation_df,
                                    width="stretch"
                                )

                            else:

                                rdkit_explain_df = explain_prediction(
                                    selected_xai_smiles
                                )

                                hybrid_explain_df = explain_hybrid_gat_prediction(
                                    selected_xai_smiles,
                                    top_n=15
                                )

                                rdkit_explain_df["Source_Model"] = "RDKit LightGBM"
                                hybrid_explain_df["Source_Model"] = "Hybrid GAT"

                                explanation_df = pd.concat(
                                    [
                                        rdkit_explain_df,
                                        hybrid_explain_df
                                    ],
                                    ignore_index=True
                                )

                                st.dataframe(
                                    explanation_df,
                                    width="stretch"
                                )

                        except Exception as explain_error:

                            st.warning(
                                f"Local explanation could not be generated: {explain_error}"
                            )

                with chart_col:

                    with st.container(border=True):

                        st.markdown("### 7️⃣ Contribution Chart")

                        try:

                            if not explanation_df.empty:

                                plot_df = explanation_df.copy()

                                if "SHAP_Value" in plot_df.columns:

                                    value_col = "SHAP_Value"

                                elif "Importance" in plot_df.columns:

                                    value_col = "Importance"

                                else:

                                    numeric_cols = plot_df.select_dtypes(
                                        include=[
                                            "number"
                                        ]
                                    ).columns.tolist()

                                    value_col = numeric_cols[0] if numeric_cols else None

                                feature_col = (
                                    "Feature"
                                    if "Feature" in plot_df.columns
                                    else plot_df.columns[0]
                                )

                                if value_col is not None:

                                    plot_df = plot_df.head(15)

                                    fig_xai, ax_xai = plt.subplots(
                                        figsize=(8, 5)
                                    )

                                    ax_xai.barh(
                                        plot_df[feature_col].astype(str)[::-1],
                                        plot_df[value_col][::-1]
                                    )

                                    ax_xai.set_xlabel(
                                        value_col
                                    )

                                    ax_xai.set_ylabel(
                                        "Descriptor / Feature"
                                    )

                                    ax_xai.set_title(
                                        "Top Local Feature Contributions"
                                    )

                                    st.pyplot(
                                        fig_xai
                                    )

                                else:

                                    st.info(
                                        "No numeric contribution column available for chart."
                                    )

                            else:

                                st.info(
                                    "Run a valid explanation to display contribution chart."
                                )

                        except Exception as chart_error:

                            st.warning(
                                f"Contribution chart could not be generated: {chart_error}"
                            )

                # ==================================================
                # GLOBAL IMPORTANCE + DRIVER INTERPRETATION
                # ==================================================

                st.markdown("---")

                driver_col1, driver_col2 = st.columns([1, 1])

                with driver_col1:

                    with st.container(border=True):

                        st.markdown("### 8️⃣ Positive vs Negative Drivers")

                        try:

                            if not explanation_df.empty:

                                driver_df = explanation_df.copy()

                                if "SHAP_Value" in driver_df.columns:

                                    driver_value_col = "SHAP_Value"

                                elif "Importance" in driver_df.columns:

                                    driver_value_col = "Importance"

                                else:

                                    numeric_cols = driver_df.select_dtypes(
                                        include=[
                                            "number"
                                        ]
                                    ).columns.tolist()

                                    driver_value_col = numeric_cols[0] if numeric_cols else None

                                feature_col = (
                                    "Feature"
                                    if "Feature" in driver_df.columns
                                    else driver_df.columns[0]
                                )

                                if driver_value_col is not None:

                                    positive_df = driver_df[
                                        driver_df[driver_value_col] > 0
                                    ].head(10)

                                    negative_df = driver_df[
                                        driver_df[driver_value_col] < 0
                                    ].head(10)

                                    st.markdown("#### Positive Drivers")

                                    if positive_df.empty:
                                        st.info("No positive drivers detected.")
                                    else:
                                        st.dataframe(
                                            positive_df[
                                                [
                                                    feature_col,
                                                    driver_value_col
                                                ]
                                            ],
                                            width="stretch"
                                        )

                                    st.markdown("#### Negative Drivers")

                                    if negative_df.empty:
                                        st.info("No negative drivers detected.")
                                    else:
                                        st.dataframe(
                                            negative_df[
                                                [
                                                    feature_col,
                                                    driver_value_col
                                                ]
                                            ],
                                            width="stretch"
                                        )

                                else:

                                    st.info(
                                        "No numeric driver column available."
                                    )

                        except Exception as driver_error:

                            st.warning(
                                f"Driver analysis could not be generated: {driver_error}"
                            )

                with driver_col2:

                    with st.container(border=True):

                        st.markdown("### 9️⃣ Global / Model Feature Importance")

                        try:

                            hybrid_importance_df = get_hybrid_feature_importance(
                                top_n=20
                            )

                            st.dataframe(
                                hybrid_importance_df,
                                width="stretch"
                            )

                            if not hybrid_importance_df.empty:

                                fig_imp, ax_imp = plt.subplots(
                                    figsize=(8, 5)
                                )

                                ax_imp.barh(
                                    hybrid_importance_df["Feature"].astype(str)[::-1],
                                    hybrid_importance_df["Importance"][::-1]
                                )

                                ax_imp.set_xlabel("Importance")
                                ax_imp.set_ylabel("Feature")
                                ax_imp.set_title("Hybrid Model Feature Importance")

                                st.pyplot(
                                    fig_imp
                                )

                        except Exception as importance_error:

                            st.warning(
                                f"Global feature importance could not be generated: {importance_error}"
                            )

                # ==================================================
                # SCIENTIFIC INTERPRETATION + DOWNLOADS
                # ==================================================

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 🔟 Scientific Interpretation")

                    narrative_outputs = build_local_xai_narrative(
                        molecule_name=selected_xai_name,
                        smiles=selected_xai_smiles,
                        rdkit_prediction=rdkit_prediction,
                        hybrid_prediction=hybrid_prediction,
                        ensemble_prediction=ensemble_prediction,
                        uncertainty_info=uncertainty_xai,
                        explanation_df=explanation_df,
                        descriptor_snapshot_df=descriptor_snapshot_df
                    )

                    top_feature_text = narrative_outputs["Top Feature"]
                    top_value_text = str(narrative_outputs["Top Value"])

                    st.markdown("#### 🧾 Local Explanation Narrative")
                    st.info(
                        narrative_outputs["Local Narrative"]
                    )

                    st.markdown("#### 🧬 Descriptor Interpretation Engine")
                    st.write(
                        narrative_outputs["Descriptor Narrative"]
                    )

                    st.markdown("#### ✅ Positive Driver Summary")

                    positive_driver_df = narrative_outputs["Positive Drivers"]

                    if positive_driver_df.empty:

                        st.info("No strong positive drivers detected.")

                    else:

                        st.dataframe(
                            positive_driver_df,
                            width="stretch",
                            hide_index=True
                        )

                    st.markdown("#### ⚠️ Negative Driver Summary")

                    negative_driver_df = narrative_outputs["Negative Drivers"]

                    if negative_driver_df.empty:

                        st.info("No strong negative drivers detected.")

                    else:

                        st.dataframe(
                            negative_driver_df,
                            width="stretch",
                            hide_index=True
                        )

                    st.markdown("#### 🔬 Auto Scientific Interpretation")

                    st.success(
                        narrative_outputs["Scientific Interpretation"]
                    )

                    interpretation_text = narrative_outputs["Scientific Interpretation"]

                st.markdown("---")

                with st.container(border=True):

                    st.markdown("### 1️⃣1️⃣ Download Explainable AI Reports")

                    summary_df = pd.DataFrame([{
                        "Molecule_Name": selected_xai_name,
                        "SMILES": selected_xai_smiles,
                        "RDKit_Prediction_K": round(rdkit_prediction, 2),
                        "Hybrid_GAT_Prediction_K": round(hybrid_prediction, 2),
                        "Ensemble_Prediction_K": round(ensemble_prediction, 2),
                        "Ensemble_Prediction_C": round(ensemble_prediction - 273.15, 2),
                        "Confidence_%": uncertainty_xai["confidence"],
                        "Confidence_Label": uncertainty_xai["confidence_label"],
                        "Top_Feature": top_feature_text,
                        "Top_Feature_Value": top_value_text,
                        "Local_Narrative": narrative_outputs.get("Local Narrative", ""),
                        "Driver_Narrative": narrative_outputs.get("Driver Narrative", ""),
                        "Descriptor_Narrative": narrative_outputs.get("Descriptor Narrative", ""),
                        "Scientific_Interpretation": interpretation_text
                    }])

                    summary_csv = summary_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download XAI Summary CSV",
                        data=summary_csv,
                        file_name="xai_summary_report.csv",
                        mime="text/csv",
                        key="download_xai_summary_narrative_csv"
                    )

                    if not explanation_df.empty:

                        explanation_csv = explanation_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Local Explanation CSV",
                            data=explanation_csv,
                            file_name="xai_local_explanation.csv",
                            mime="text/csv",
                            key="download_xai_local_explanation_csv"
                        )

                    if not hybrid_importance_df.empty:

                        importance_csv = hybrid_importance_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Global Feature Importance CSV",
                            data=importance_csv,
                            file_name="xai_global_feature_importance.csv",
                            mime="text/csv",
                            key="download_xai_global_importance_csv"
                        )

        except Exception as e:

            st.error(
                f"Explainable AI analysis failed: {e}"
            )



    # ==================================================
    # TAB 15 — ABOUT PLATFORM
    # ==================================================

    with tab15:

        st.subheader("🏛️ About Platform")

        try:
            about_df = load_molecule_dataset()
            about_molecule_count = len(about_df)
        except Exception:
            about_molecule_count = "N/A"

        st.info(
            "Enterprise-grade molecular AI platform for melting point prediction, "
            "explainable AI, chemical-space analytics, scaffold exploration, "
            "OOD reliability analysis, and drug-likeness discovery."
        )

        card1, card2, card3, card4 = st.columns(4)

        with card1:
            with st.container(border=True):
                st.markdown("### 🧠 Hybrid AI")
                st.write(
                    "RDKit descriptors, LightGBM, GNN ensemble learning, "
                    "and uncertainty estimation."
                )

        with card2:
            with st.container(border=True):
                st.markdown("### 🧬 Chemical Space")
                st.write(
                    "PCA, t-SNE, UMAP, scaffold analysis, similarity search, "
                    "and molecular clustering."
                )

        with card3:
            with st.container(border=True):
                st.markdown("### 🔬 Explainable AI")
                st.write(
                    "SHAP interpretability, descriptor importance, prediction "
                    "transparency, and contribution analysis."
                )

        with card4:
            with st.container(border=True):
                st.markdown("### 🚀 Deployment")
                st.write(
                    "Streamlit deployment, PDF reporting, enterprise workflows, "
                    "and scalable AI architecture."
                )

        st.markdown("---")

        overview_col, stats_col = st.columns([0.62, 0.38])

        with overview_col:

            with st.container(border=True):

                st.markdown("### ⚙️ Core Platform Modules")

                modules_df = pd.DataFrame({
                    "Module": [
                        "Molecular Melting Point Prediction",
                        "3D Molecular Visualization",
                        "OOD Reliability Detection",
                        "Chemical Space PCA / t-SNE / UMAP",
                        "Scaffold & Similarity Analysis",
                        "Explainable AI",
                        "Drug-Likeness Discovery",
                        "Enterprise PDF Reporting"
                    ],
                    "Purpose": [
                        "Predict molecular melting point",
                        "Visualize molecules interactively",
                        "Assess prediction reliability",
                        "Explore molecular chemical space",
                        "Analyze molecular cores and similarity",
                        "Explain model decisions",
                        "Screen drug-like properties",
                        "Generate branded downloadable reports"
                    ]
                })

                st.dataframe(
                    modules_df,
                    width="stretch",
                    hide_index=True
                )

            with st.container(border=True):

                st.markdown("### 🧪 Technologies Used")

                tech_df = pd.DataFrame({
                    "Technology": [
                        "Python",
                        "Streamlit",
                        "RDKit",
                        "LightGBM",
                        "PyTorch Geometric",
                        "SHAP",
                        "Plotly",
                        "Scikit-learn",
                        "Pandas",
                        "NumPy",
                        "ReportLab"
                    ],
                    "Role": [
                        "Core programming",
                        "Interactive web app",
                        "Molecular descriptors and chemistry",
                        "Machine learning model",
                        "Graph neural network support",
                        "Explainable AI",
                        "Interactive visualization",
                        "ML utilities",
                        "Data processing",
                        "Numerical computing",
                        "PDF report generation"
                    ]
                })

                st.dataframe(
                    tech_df,
                    width="stretch",
                    hide_index=True
                )

        with stats_col:

            with st.container(border=True):

                st.markdown("### 📊 Platform Snapshot")

                st.metric("Dataset Size", about_molecule_count)
                st.metric("Platform Tabs", "15")
                st.metric("Visualization Modules", "PCA / t-SNE / UMAP")
                st.metric("Explainability", "SHAP AI")
                st.metric("Report System", "PDF + CSV")

            with st.container(border=True):

                st.markdown("### 🏷️ Research Identity")

                st.write(
                    "Hybrid GNN AI Cheminformatics Platform is designed as a "
                    "research-grade and deployment-ready molecular AI workspace."
                )

                st.success(
                    "Built for cheminformatics experimentation, explainable prediction, "
                    "model reliability assessment, and portfolio-grade AI deployment."
                )

        st.markdown("---")

        st.caption(
            "Hybrid GNN AI Cheminformatics Platform · Enterprise Molecular AI Workspace"
        )



    # ==================================================
    # TAB 16 — ADMIN USERS
    # ==================================================

    with tab16:

        st.subheader("👥 Admin Users")

        if not is_admin_user():

            st.warning(
                "Access denied. This section is available only for admin users."
            )

        else:

            st.markdown(
                """
                <div style="
                    background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                    border: 1px solid #c4b5fd;
                    border-radius: 16px;
                    padding: 14px 16px;
                    margin-bottom: 18px;
                    color: #312e81;
                    font-weight: 650;
                ">
                Admin control panel: view registered users, export user list, update roles, delete inactive users, and securely reset passwords. Passwords are never displayed.
                </div>
                """,
                unsafe_allow_html=True
            )

            users_df = load_registered_users()

            with st.container(border=True):

                st.markdown("### 1️⃣ Registered Users Summary")

                col_u1, col_u2, col_u3, col_u4 = st.columns(4)

                total_users = len(users_df)
                admin_count = int((users_df["Role"] == "admin").sum()) if not users_df.empty else 0
                normal_user_count = int((users_df["Role"] == "user").sum()) if not users_df.empty else 0

                with col_u1:
                    st.metric("Total Users", total_users)

                with col_u2:
                    st.metric("Admins", admin_count)

                with col_u3:
                    st.metric("Regular Users", normal_user_count)

                with col_u4:
                    st.metric("Current User", get_current_username())

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 2️⃣ Registered Users Table")

                if users_df.empty:

                    st.info("No registered users found.")

                else:

                    display_paginated_dataframe(
                        df=users_df,
                        table_key="admin_registered_users_table",
                        rows_per_page=100
                    )

                    users_csv = users_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Registered Users CSV",
                        data=users_csv,
                        file_name="registered_users_safe_export.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "Security note: real passwords are not stored or displayed. Only hashed passwords and salts exist inside the SQLite database."
                    )

            st.markdown("---")

            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:

                with st.container(border=True):

                    st.markdown("### 4️⃣ Update User Role")

                    if users_df.empty:

                        st.info("No users available.")

                    else:

                        selected_role_username = st.selectbox(
                            "Select user",
                            users_df["Username"].tolist(),
                            key="admin_role_user_select"
                        )

                        selected_new_role = st.selectbox(
                            "New role",
                            [
                                "user",
                                "admin"
                            ],
                            key="admin_new_role_select"
                        )

                        confirm_role_update = st.checkbox(
                            "I confirm updating this user's role",
                            key="confirm_admin_role_update"
                        )

                        if st.button(
                            "Update Role",
                            key="admin_update_user_role_button"
                        ):

                            if confirm_role_update:

                                update_user_role(
                                    selected_role_username,
                                    selected_new_role
                                )

                                st.success(
                                    f"Role updated for {selected_role_username}."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "Please confirm before updating the role."
                                )

            with action_col2:

                with st.container(border=True):

                    st.markdown("### 5️⃣ Delete User")

                    if users_df.empty:

                        st.info("No users available.")

                    else:

                        selected_delete_username = st.selectbox(
                            "Select user to delete",
                            users_df["Username"].tolist(),
                            key="admin_delete_user_select"
                        )

                        confirm_user_delete = st.checkbox(
                            "I confirm deleting this user account",
                            key="confirm_admin_user_delete"
                        )

                        if st.button(
                            "Delete User",
                            key="admin_delete_user_button"
                        ):

                            if confirm_user_delete:

                                success, message = delete_registered_user(
                                    selected_delete_username
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.warning(message)

                            else:

                                st.warning(
                                    "Please confirm before deleting the user."
                                )

            with action_col3:

                with st.container(border=True):

                    st.markdown("### 6️⃣ Reset User Password")

                    if users_df.empty:

                        st.info("No users available.")

                    else:

                        selected_reset_username = st.selectbox(
                            "Select user for password reset",
                            users_df["Username"].tolist(),
                            key="admin_reset_password_user_select"
                        )

                        new_admin_password = st.text_input(
                            "New password",
                            type="password",
                            key="admin_reset_new_password"
                        )

                        confirm_new_admin_password = st.text_input(
                            "Confirm new password",
                            type="password",
                            key="admin_reset_confirm_password"
                        )

                        confirm_password_reset = st.checkbox(
                            "I confirm resetting this user's password",
                            key="confirm_admin_password_reset"
                        )

                        if st.button(
                            "Reset Password",
                            key="admin_reset_user_password_button"
                        ):

                            if not confirm_password_reset:

                                st.warning(
                                    "Please confirm before resetting the password."
                                )

                            elif new_admin_password != confirm_new_admin_password:

                                st.error(
                                    "Passwords do not match."
                                )

                            else:

                                success, message = reset_user_password(
                                    selected_reset_username,
                                    new_admin_password
                                )

                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)

                        st.info(
                            "Security note: old passwords are never displayed. Resetting creates a new salted password hash."
                        )



    # ==================================================
    # TAB 17 — DATAOPS / ENTERPRISE DATABASE
    # ==================================================

    with tab17:

        st.subheader("🗄️ DataOps / Enterprise Database")

        if not is_admin_user():

            st.warning(
                "Access denied. This section is available only for admin users."
            )

        else:

            st.markdown(
                """
                <div style="
                    background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                    border: 1px solid #c4b5fd;
                    border-radius: 16px;
                    padding: 14px 16px;
                    margin-bottom: 18px;
                    color: #312e81;
                    font-weight: 650;
                ">
                Enterprise DataOps Console: persistent storage for predictions, SHAP outputs,
                uploaded CSV metadata, OOD results, report metadata, and user activity logs.
                </div>
                """,
                unsafe_allow_html=True
            )

            storage_summary_df = build_enterprise_storage_summary()

            with st.container(border=True):

                st.markdown("### 1️⃣ Enterprise Storage Summary")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                total_tables = len(storage_summary_df)

                total_rows = int(
                    storage_summary_df["Rows"].sum()
                ) if not storage_summary_df.empty else 0

                db_exists = os.path.exists(ENTERPRISE_DB_PATH)

                db_size_mb = (
                    round(os.path.getsize(ENTERPRISE_DB_PATH) / (1024 * 1024), 4)
                    if db_exists
                    else 0
                )

                with col_s1:
                    st.metric("Enterprise Tables", total_tables)

                with col_s2:
                    st.metric("Stored Records", total_rows)

                with col_s3:
                    st.metric("Database Size MB", db_size_mb)

                with col_s4:
                    st.metric("Database", ENTERPRISE_DB_PATH)

                st.dataframe(
                    storage_summary_df,
                    width="stretch",
                    hide_index=True
                )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 2️⃣ Inspect Enterprise Tables")

                enterprise_tables = storage_summary_df["Table"].tolist()

                selected_enterprise_table = st.selectbox(
                    "Select enterprise table",
                    enterprise_tables,
                    key="enterprise_dataops_selected_table"
                )

                selected_enterprise_limit = st.number_input(
                    "Preview row limit",
                    min_value=10,
                    max_value=5000,
                    value=500,
                    step=50,
                    key="enterprise_dataops_limit"
                )

                try:

                    enterprise_preview_df = load_enterprise_table(
                        selected_enterprise_table,
                        limit=selected_enterprise_limit
                    )

                    if enterprise_preview_df.empty:
                        st.info("No records available in this table yet.")
                    else:
                        st.dataframe(
                            enterprise_preview_df,
                            width="stretch",
                            hide_index=True
                        )

                    enterprise_csv = enterprise_preview_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Selected Enterprise Table CSV",
                        data=enterprise_csv,
                        file_name=f"{selected_enterprise_table}.csv",
                        mime="text/csv",
                        key="enterprise_dataops_table_download"
                    )

                except Exception as e:

                    st.error(
                        f"Could not load selected enterprise table: {e}"
                    )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Enterprise Database Backup")

                if os.path.exists(ENTERPRISE_DB_PATH):

                    with open(ENTERPRISE_DB_PATH, "rb") as db_file:
                        enterprise_db_bytes = db_file.read()

                    st.download_button(
                        label="Download Enterprise AI Storage DB Backup",
                        data=enterprise_db_bytes,
                        file_name=ENTERPRISE_DB_PATH,
                        mime="application/octet-stream",
                        key="enterprise_dataops_db_backup_download"
                    )

                    st.success(
                        "Enterprise database backup is ready."
                    )

                else:

                    st.warning(
                        "Enterprise database file not found yet."
                    )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 4️⃣ Storage Coverage")

                coverage_df = pd.DataFrame({
                    "Storage Area": [
                        "Predictions",
                        "SHAP / XAI Outputs",
                        "Uploaded CSV Metadata",
                        "OOD Detection Results",
                        "Report Metadata",
                        "User Activity Logs"
                    ],
                    "Database Table": [
                        "enterprise_predictions",
                        "enterprise_shap_outputs",
                        "enterprise_uploaded_csvs",
                        "enterprise_ood_results",
                        "enterprise_report_metadata",
                        "enterprise_user_activity"
                    ],
                    "Status": [
                        "Enabled",
                        "Enabled",
                        "Enabled",
                        "Enabled",
                        "Enabled",
                        "Enabled"
                    ]
                })

                st.dataframe(
                    coverage_df,
                    width="stretch",
                    hide_index=True
                )



    # ==================================================
    # TAB 18 — MODEL MONITORING
    # ==================================================

    with tab18:

        st.subheader("📡 Model Monitoring")

        if not is_admin_user():

            st.warning(
                "Access denied. This section is available only for admin users."
            )

        else:

            st.markdown(
                """
                <div style="
                    background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                    border: 1px solid #c4b5fd;
                    border-radius: 16px;
                    padding: 14px 16px;
                    margin-bottom: 18px;
                    color: #312e81;
                    font-weight: 650;
                ">
                Model monitoring layer: prediction drift, confidence degradation,
                OOD trend tracking, failure monitoring, and model health dashboard.
                </div>
                """,
                unsafe_allow_html=True
            )

            predictions_monitor_df, ood_monitor_df, activity_monitor_df = load_model_monitoring_data()

            health_info = calculate_model_health_status(
                predictions_monitor_df,
                ood_monitor_df
            )

            summary_monitor_df = build_model_monitoring_summary(
                predictions_monitor_df,
                ood_monitor_df
            )

            with st.container(border=True):

                st.markdown("### 1️⃣ Model Health Summary")

                h1, h2, h3 = st.columns(3)

                with h1:
                    st.metric(
                        "Health Status",
                        health_info["Health Status"]
                    )

                with h2:
                    st.metric(
                        "Risk Level",
                        health_info["Risk Level"]
                    )

                with h3:
                    st.metric(
                        "Stored Predictions",
                        len(predictions_monitor_df)
                    )

                st.info(
                    health_info["Recommendation"]
                )

                st.dataframe(
                    summary_monitor_df,
                    width="stretch",
                    hide_index=True
                )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 2️⃣ Prediction Drift Tracking")

                if predictions_monitor_df.empty:

                    st.info(
                        "No prediction records available yet. Run predictions to activate drift tracking."
                    )

                else:

                    drift_df = (
                        predictions_monitor_df
                        .dropna(subset=["date"])
                        .groupby("date")
                        .agg(
                            Average_Prediction_K=("predicted_kelvin", "mean"),
                            Prediction_Count=("id", "count")
                        )
                        .reset_index()
                    )

                    if drift_df.empty:

                        st.info("Not enough dated prediction records for drift tracking.")

                    else:

                        st.line_chart(
                            drift_df.set_index("date")[
                                [
                                    "Average_Prediction_K",
                                    "Prediction_Count"
                                ]
                            ]
                        )

                        st.dataframe(
                            drift_df,
                            width="stretch",
                            hide_index=True
                        )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 3️⃣ Confidence Degradation Monitoring")

                if predictions_monitor_df.empty or "confidence" not in predictions_monitor_df.columns:

                    st.info("No confidence records available yet.")

                else:

                    confidence_df = (
                        predictions_monitor_df
                        .dropna(subset=["date"])
                        .groupby("date")
                        .agg(
                            Average_Confidence=("confidence", "mean"),
                            Average_Uncertainty=("uncertainty", "mean")
                        )
                        .reset_index()
                    )

                    if confidence_df.empty:

                        st.info("No confidence trend available yet.")

                    else:

                        st.line_chart(
                            confidence_df.set_index("date")
                        )

                        low_confidence_df = predictions_monitor_df[
                            predictions_monitor_df["confidence"].fillna(100) < 70
                        ].copy()

                        st.warning(
                            f"Low-confidence prediction count: {len(low_confidence_df)}"
                        )

                        if not low_confidence_df.empty:

                            st.dataframe(
                                low_confidence_df[
                                    [
                                        "created_at",
                                        "username",
                                        "molecule_name",
                                        "smiles",
                                        "model_used",
                                        "confidence",
                                        "confidence_label",
                                        "uncertainty"
                                    ]
                                ],
                                width="stretch",
                                hide_index=True
                            )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 4️⃣ OOD / Reliability Trend")

                if ood_monitor_df.empty:

                    st.info(
                        "No OOD results stored yet."
                    )

                else:

                    ood_summary_df = (
                        ood_monitor_df
                        .groupby("ood_status")
                        .size()
                        .reset_index(name="Count")
                    )

                    st.bar_chart(
                        ood_summary_df.set_index("ood_status")
                    )

                    ood_daily_df = (
                        ood_monitor_df
                        .dropna(subset=["date"])
                        .groupby(["date", "ood_status"])
                        .size()
                        .reset_index(name="Count")
                    )

                    st.dataframe(
                        ood_daily_df,
                        width="stretch",
                        hide_index=True
                    )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 5️⃣ Error / Failure Monitoring")

                if predictions_monitor_df.empty:

                    st.info("No prediction records available.")

                else:

                    failure_df = predictions_monitor_df[
                        predictions_monitor_df["status"].astype(str).str.contains(
                            "Failed",
                            case=False,
                            na=False
                        )
                    ].copy()

                    f1, f2 = st.columns(2)

                    with f1:
                        st.metric(
                            "Failed Predictions",
                            len(failure_df)
                        )

                    with f2:
                        st.metric(
                            "Failure Rate %",
                            round(
                                len(failure_df) / len(predictions_monitor_df) * 100,
                                2
                            ) if len(predictions_monitor_df) > 0 else 0
                        )

                    if failure_df.empty:

                        st.success("No failed predictions recorded.")

                    else:

                        st.dataframe(
                            failure_df[
                                [
                                    "created_at",
                                    "username",
                                    "smiles",
                                    "model_used",
                                    "status"
                                ]
                            ],
                            width="stretch",
                            hide_index=True
                        )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 6️⃣ Scaffold Drift Proxy")

                st.info(
                    "Scaffold drift proxy currently uses molecular SMILES activity distribution. "
                    "A deeper Murcko scaffold drift layer can be added later using stored scaffold fingerprints."
                )

                if predictions_monitor_df.empty:

                    st.info("No prediction records available.")

                else:

                    top_smiles_df = (
                        predictions_monitor_df
                        .groupby("smiles")
                        .size()
                        .reset_index(name="Prediction Count")
                        .sort_values(
                            by="Prediction Count",
                            ascending=False
                        )
                        .head(20)
                    )

                    st.dataframe(
                        top_smiles_df,
                        width="stretch",
                        hide_index=True
                    )

            st.markdown("---")

            with st.container(border=True):

                st.markdown("### 7️⃣ Export Model Monitoring Report")

                export_summary = pd.concat(
                    [
                        summary_monitor_df.assign(Section="Model Summary")
                    ],
                    ignore_index=True
                )

                export_csv = export_summary.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Model Monitoring Summary CSV",
                    data=export_csv,
                    file_name="model_monitoring_summary.csv",
                    mime="text/csv",
                    key="model_monitoring_summary_download"
                )



    # ==================================================
    # TAB 19 — MOLECULE COMPARISON WORKSPACE
    # ==================================================

    with tab19:

        st.subheader("⚖️ Molecule Comparison Workspace")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Compare Molecule A vs Molecule B using structural similarity, RDKit descriptors,
            scaffold matching, SHAP explanations, predicted melting point difference,
            and an overall molecular similarity percentage score.
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            comparison_df = load_molecule_dataset()

            total_dataset_rows = len(comparison_df)

            try:
                unique_molecules = comparison_df["SMILES"].astype(str).nunique()
            except Exception:
                unique_molecules = total_dataset_rows

            try:
                valid_smiles_count = comparison_df["SMILES"].dropna().astype(str).nunique()
            except Exception:
                valid_smiles_count = "N/A"

            try:
                comparison_scaffold_count = comparison_df["SMILES"].apply(
                    get_murcko_scaffold
                ).nunique()
            except Exception:
                comparison_scaffold_count = "N/A"

            st.markdown("### 🧪 Molecule Comparison Dataset Coverage")

            coverage_col1, coverage_col2, coverage_col3, coverage_col4 = st.columns(4)

            with coverage_col1:
                st.metric(
                    "Unique Molecules",
                    unique_molecules
                )

            with coverage_col2:
                st.metric(
                    "Total Dataset Rows",
                    total_dataset_rows
                )

            with coverage_col3:
                st.metric(
                    "Valid SMILES",
                    valid_smiles_count
                )

            with coverage_col4:
                st.metric(
                    "Unique Scaffolds",
                    comparison_scaffold_count
                )

            st.info(
                f"This workspace compares molecules from a dataset containing "
                f"{unique_molecules} unique molecules across {total_dataset_rows} total dataset rows."
            )

            st.markdown("---")

            st.markdown("### 1️⃣ Select Molecules")

            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:

                st.markdown("#### Molecule A")

                molecule_a_choice = st.selectbox(
                    "Select Molecule A",
                    comparison_df["Molecule_Display"].tolist(),
                    key="comparison_molecule_a_select"
                )

                smiles_a = comparison_df.loc[
                    comparison_df["Molecule_Display"] == molecule_a_choice,
                    "SMILES"
                ].iloc[0]

                name_a = comparison_df.loc[
                    comparison_df["Molecule_Display"] == molecule_a_choice,
                    "Molecule_Name"
                ].iloc[0]

                st.code(smiles_a)

            with comp_col2:

                st.markdown("#### Molecule B")

                molecule_b_choice = st.selectbox(
                    "Select Molecule B",
                    comparison_df["Molecule_Display"].tolist(),
                    key="comparison_molecule_b_select"
                )

                smiles_b = comparison_df.loc[
                    comparison_df["Molecule_Display"] == molecule_b_choice,
                    "SMILES"
                ].iloc[0]

                name_b = comparison_df.loc[
                    comparison_df["Molecule_Display"] == molecule_b_choice,
                    "Molecule_Name"
                ].iloc[0]

                st.code(smiles_b)

            if st.button(
                "Run Molecule Comparison",
                key="run_molecule_comparison_button"
            ):

                st.markdown("---")

                with st.spinner("Running molecule comparison..."):

                    desc_a = calculate_basic_rdkit_descriptor_dict(smiles_a)
                    desc_b = calculate_basic_rdkit_descriptor_dict(smiles_b)

                    structural_similarity = calculate_structural_similarity_percent(
                        smiles_a,
                        smiles_b
                    )

                    descriptor_similarity = calculate_descriptor_similarity_percent(
                        desc_a,
                        desc_b
                    )

                    scaffold_match, scaffold_a, scaffold_b = calculate_scaffold_match_percent(
                        smiles_a,
                        smiles_b
                    )

                    chemical_space_label, chemical_space_score = calculate_chemical_space_distance_score(
                        descriptor_similarity
                    )

                    pred_a_raw = predict_melting_point(smiles_a)
                    pred_b_raw = predict_melting_point(smiles_b)

                    rdkit_pred_a = float(pred_a_raw[0] if isinstance(pred_a_raw, (list, tuple, np.ndarray)) else pred_a_raw)
                    rdkit_pred_b = float(pred_b_raw[0] if isinstance(pred_b_raw, (list, tuple, np.ndarray)) else pred_b_raw)

                    melting_point_difference = abs(rdkit_pred_a - rdkit_pred_b)

                    shap_a = pd.DataFrame()
                    shap_b = pd.DataFrame()
                    shap_similarity = None

                    try:
                        shap_a = explain_prediction(smiles_a)
                        shap_b = explain_prediction(smiles_b)

                        shap_similarity = calculate_shap_similarity_percent(
                            shap_a,
                            shap_b
                        )

                    except Exception:
                        shap_similarity = None

                    overall_similarity = calculate_overall_molecule_similarity_score(
                        structural_similarity=structural_similarity,
                        descriptor_similarity=descriptor_similarity,
                        scaffold_match=scaffold_match,
                        chemical_space_score=chemical_space_score,
                        shap_similarity=shap_similarity
                    )

                st.success("Molecule comparison completed.")

                st.markdown("### 2️⃣ Comparison Scorecard")

                score_col1, score_col2, score_col3, score_col4 = st.columns(4)

                with score_col1:
                    st.metric("Structural Similarity", f"{structural_similarity}%")

                with score_col2:
                    st.metric("Descriptor Similarity", f"{descriptor_similarity}%")

                with score_col3:
                    st.metric("Scaffold Match", f"{scaffold_match}%")

                with score_col4:
                    st.metric("Overall Similarity", f"{overall_similarity}%")

                score_col5, score_col6, score_col7, score_col8 = st.columns(4)

                with score_col5:
                    st.metric(
                        "SHAP Similarity",
                        f"{shap_similarity}%" if shap_similarity is not None else "N/A"
                    )

                with score_col6:
                    st.metric("Chemical Space", chemical_space_label)

                with score_col7:
                    st.metric("MP Difference", f"{round(melting_point_difference, 2)} K")

                with score_col8:
                    st.metric("Chemical Space Score", f"{chemical_space_score}%")

                st.markdown("---")

                st.markdown("### 3️⃣ Prediction & Scaffold Comparison")

                prediction_comparison_df = pd.DataFrame({
                    "Metric": [
                        "Molecule Name",
                        "SMILES",
                        "Predicted Melting Point K",
                        "Predicted Melting Point C",
                        "Murcko Scaffold"
                    ],
                    "Molecule A": [
                        name_a,
                        smiles_a,
                        round(rdkit_pred_a, 2),
                        round(rdkit_pred_a - 273.15, 2),
                        scaffold_a
                    ],
                    "Molecule B": [
                        name_b,
                        smiles_b,
                        round(rdkit_pred_b, 2),
                        round(rdkit_pred_b - 273.15, 2),
                        scaffold_b
                    ]
                })

                st.dataframe(prediction_comparison_df, width="stretch", hide_index=True)

                st.markdown("---")

                st.markdown("### 4️⃣ RDKit Descriptor Comparison")

                descriptor_comparison_df = build_molecule_comparison_dataframe(desc_a, desc_b)

                st.dataframe(descriptor_comparison_df, width="stretch", hide_index=True)

                st.markdown("---")

                st.markdown("### 5️⃣ 2D Structure Comparison")

                img_col1, img_col2 = st.columns(2)

                mol_a = Chem.MolFromSmiles(smiles_a)
                mol_b = Chem.MolFromSmiles(smiles_b)

                with img_col1:
                    st.markdown(f"#### Molecule A: {name_a}")
                    if mol_a is not None:
                        st.image(Draw.MolToImage(mol_a, size=(350, 300)))

                with img_col2:
                    st.markdown(f"#### Molecule B: {name_b}")
                    if mol_b is not None:
                        st.image(Draw.MolToImage(mol_b, size=(350, 300)))

                st.markdown("---")

                st.markdown("### 6️⃣ 3D Structure Comparison")

                st.info(
                    "Interactive 3D comparison uses the existing py3Dmol molecular viewer. "
                    "If true 3D conformer generation is not possible, the app will show the safe fallback viewer."
                )

                view3d_col1, view3d_col2 = st.columns(2)

                with view3d_col1:

                    st.markdown(f"#### Molecule A 3D: {name_a}")

                    try:
                        show_3d_molecule(
                            smiles_a,
                            width=520,
                            height=420,
                            viewer_key="comparison_molecule_a_3d"
                        )
                    except Exception as e:
                        st.warning(f"3D viewer unavailable for Molecule A: {e}")

                with view3d_col2:

                    st.markdown(f"#### Molecule B 3D: {name_b}")

                    try:
                        show_3d_molecule(
                            smiles_b,
                            width=520,
                            height=420,
                            viewer_key="comparison_molecule_b_3d"
                        )
                    except Exception as e:
                        st.warning(f"3D viewer unavailable for Molecule B: {e}")

                st.markdown("---")

                st.markdown("### 7️⃣ SHAP Comparison")

                if shap_similarity is None:

                    st.info("SHAP comparison was not available for this pair.")

                else:

                    shap_merge_df = pd.merge(
                        shap_a[["Feature", "SHAP_Value"]],
                        shap_b[["Feature", "SHAP_Value"]],
                        on="Feature",
                        suffixes=("_A", "_B")
                    )

                    shap_merge_df["Abs_Difference"] = (
                        shap_merge_df["SHAP_Value_A"] - shap_merge_df["SHAP_Value_B"]
                    ).abs()

                    st.dataframe(
                        shap_merge_df.sort_values(by="Abs_Difference", ascending=False),
                        width="stretch",
                        hide_index=True
                    )

                st.markdown("---")

                st.markdown("### 8️⃣ Export Comparison Results")

                scorecard_df = pd.DataFrame({
                    "Score": [
                        "Structural Similarity %",
                        "Descriptor Similarity %",
                        "Scaffold Match %",
                        "SHAP Similarity %",
                        "Chemical Space Score",
                        "Overall Similarity %",
                        "Melting Point Difference K",
                        "Molecule A",
                        "Molecule B"
                    ],
                    "Value": [
                        structural_similarity,
                        descriptor_similarity,
                        scaffold_match,
                        shap_similarity if shap_similarity is not None else "N/A",
                        chemical_space_score,
                        overall_similarity,
                        round(melting_point_difference, 2),
                        name_a,
                        name_b
                    ]
                })

                export_csv = scorecard_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Molecule Comparison CSV",
                    data=export_csv,
                    file_name="molecule_comparison_scorecard.csv",
                    mime="text/csv",
                    key="download_molecule_comparison_scorecard"
                )

                try:
                    save_enterprise_user_activity(
                        "Molecule Comparison",
                        f"Compared {smiles_a} vs {smiles_b}; overall similarity {overall_similarity}%"
                    )
                except Exception:
                    pass

        except Exception as e:

            st.error(f"Molecule comparison workspace failed: {e}")



    # ==================================================
    # TAB 20 — SCIENTIFIC BENCHMARK
    # ==================================================

    with tab20:

        st.subheader("🏆 Scientific Benchmark")

        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #ede9fe 0%, #eef2ff 100%);
                border: 1px solid #c4b5fd;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 18px;
                color: #312e81;
                font-weight: 650;
            ">
            Academic benchmark section: model leaderboard, MAE/RMSE/R² comparison,
            hybrid vs ML vs GNN interpretation, scaffold split validation guidance,
            and exportable validation statistics.
            </div>
            """,
            unsafe_allow_html=True
        )

        benchmark_df, benchmark_source = load_scientific_benchmark_results()
        ranked_benchmark_df = rank_benchmark_models(benchmark_df)
        validation_stats_df = build_validation_statistics_table(benchmark_df)
        scaffold_split_df = build_scaffold_split_template()
        benchmark_interpretation = build_benchmark_interpretation(benchmark_df)

        with st.container(border=True):

            st.markdown("### 1️⃣ Benchmark Overview")

            b1, b2, b3, b4 = st.columns(4)

            with b1:
                st.metric(
                    "Models Compared",
                    len(benchmark_df)
                )

            with b2:
                st.metric(
                    "Benchmark Source",
                    benchmark_source
                )

            with b3:
                st.metric(
                    "Primary Metric",
                    "MAE"
                )

            with b4:
                st.metric(
                    "Validation Focus",
                    "Scaffold Robustness"
                )

            st.info(
                "This benchmark tab first looks for benchmark_results.csv or scientific_benchmark.csv. "
                "If no benchmark file is found, it automatically loads the available project validation summary "
                "and app prediction log count. For complete academic benchmarking, add a CSV with true validation metrics."
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 🔎 Benchmark Data Source Note")

            st.write(
                "Normal app predictions are stored in prediction_logs.db / enterprise_ai_storage.db. "
                "Those records show app usage and prediction history, but they do not automatically provide MAE, RMSE, or R² "
                "unless true experimental melting point values are available for comparison."
            )

            st.write(
                "The leaderboard below uses a benchmark CSV if present. If no CSV exists, it uses the available project "
                "validation summary and displays app prediction log counts for traceability."
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 2️⃣ Model Benchmark Leaderboard")

            st.dataframe(
                ranked_benchmark_df,
                width="stretch",
                hide_index=True
            )

            if "MAE" in ranked_benchmark_df.columns and ranked_benchmark_df["MAE"].notna().any():

                chart_df = ranked_benchmark_df.dropna(
                    subset=["MAE"]
                ).copy()

                chart_df = chart_df.set_index("Model")

                st.bar_chart(
                    chart_df[["MAE"]]
                )

            else:

                st.warning(
                    "MAE values are not available yet. Add benchmark_results.csv to activate the leaderboard chart."
                )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 3️⃣ MAE / RMSE / R² Comparison")

            metric_cols = [
                col for col in [
                    "Model",
                    "MAE",
                    "RMSE",
                    "R2"
                ]
                if col in ranked_benchmark_df.columns
            ]

            metric_df = ranked_benchmark_df[
                metric_cols
            ].copy()

            st.dataframe(
                metric_df,
                width="stretch",
                hide_index=True
            )

            numeric_metric_cols = [
                col for col in [
                    "MAE",
                    "RMSE",
                    "R2"
                ]
                if col in metric_df.columns
            ]

            if numeric_metric_cols and "Model" in metric_df.columns:

                plot_metric_df = metric_df.copy()

                for metric_col in numeric_metric_cols:
                    plot_metric_df[metric_col] = pd.to_numeric(
                        plot_metric_df[metric_col],
                        errors="coerce"
                    )

                plot_metric_df = plot_metric_df.set_index("Model")

                st.line_chart(
                    plot_metric_df[numeric_metric_cols]
                )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 4️⃣ Hybrid vs ML vs GNN Comparison")

            comparison_table = pd.DataFrame({
                "Model Family": [
                    "Classical ML",
                    "Hybrid Descriptor + GNN/GAT",
                    "Ensemble AI"
                ],
                "Expected Strength": [
                    "Fast, interpretable descriptor-based predictions",
                    "Learns molecular graph patterns plus descriptors",
                    "Combines strengths and reduces single-model risk"
                ],
                "Scientific Limitation": [
                    "May miss graph/topology nuance",
                    "Can require more compute and careful validation",
                    "May hide individual model weaknesses"
                ],
                "Recommended Use": [
                    "Baseline and deployment stability",
                    "Advanced chemistry-aware prediction",
                    "Final robust prediction layer"
                ]
            })

            st.dataframe(
                comparison_table,
                width="stretch",
                hide_index=True
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 5️⃣ Scaffold Split Results / Validation Design")

            st.dataframe(
                scaffold_split_df,
                width="stretch",
                hide_index=True
            )

            st.info(
                "Scaffold split validation is academically stronger than random split because it checks "
                "whether the model generalizes to new molecular cores rather than memorizing similar compounds."
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 6️⃣ Validation Statistics")

            st.dataframe(
                validation_stats_df,
                width="stretch",
                hide_index=True
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 7️⃣ Academic Interpretation")

            st.success(
                benchmark_interpretation
            )

            st.markdown(
                """
                **Recommended reporting language:**  
                The benchmark evaluates melting point prediction performance using MAE, RMSE, and R².  
                MAE is emphasized because it is directly interpretable in Kelvin and less sensitive to extreme
                squared errors than RMSE. Scaffold-aware validation is recommended to test chemical
                generalization beyond random molecule splits.
                """
            )

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### 8️⃣ Export Benchmark Report")

            benchmark_export_df = ranked_benchmark_df.copy()

            benchmark_csv = benchmark_export_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Scientific Benchmark CSV",
                data=benchmark_csv,
                file_name="scientific_benchmark_leaderboard.csv",
                mime="text/csv",
                key="download_scientific_benchmark_csv"
            )

            validation_csv = validation_stats_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Validation Statistics CSV",
                data=validation_csv,
                file_name="validation_statistics.csv",
                mime="text/csv",
                key="download_validation_statistics_csv"
            )

            benchmark_template_df = pd.DataFrame({
                "Model": [
                    "RDKit LightGBM",
                    "Hybrid Descriptor + GAT",
                    "Ensemble AI"
                ],
                "Model_Type": [
                    "Classical ML",
                    "Hybrid AI / GNN",
                    "Weighted Ensemble"
                ],
                "Validation_Strategy": [
                    "Holdout / Scaffold Split",
                    "Holdout / Scaffold Split",
                    "Holdout / Scaffold Split"
                ],
                "MAE": [
                    "",
                    "",
                    ""
                ],
                "RMSE": [
                    "",
                    "",
                    ""
                ],
                "R2": [
                    "",
                    "",
                    ""
                ],
                "Scientific_Role": [
                    "Descriptor-based baseline",
                    "Graph-aware hybrid model",
                    "Final robust ensemble"
                ]
            })

            benchmark_template_csv = benchmark_template_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Benchmark Results Template CSV",
                data=benchmark_template_csv,
                file_name="benchmark_results.csv",
                mime="text/csv",
                key="download_benchmark_results_template_csv"
            )


    st.markdown("---")

    st.caption(
        "Hybrid GNN AI Cheminformatics Platform | "
        "Enhanced PDF Report + Batch Confidence Report + Dashboard Summary + "
        "Murcko Scaffold Analysis + OOD Detection + Chemical Space PCA + t-SNE + UMAP + Interactive Plotly UMAP + AI Overlay + Drug-Likeness Analysis + Explainable AI + About Platform + Admin Users + Password Reset + Admin Monitoring"
    )


elif st.session_state["authentication_status"] is False:

    st.error("Incorrect username or password")

else:
    st.warning("Please enter username and password")
