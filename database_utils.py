import sqlite3
from datetime import datetime


DB_NAME = "prediction_logs.db"


# =====================================================
# CREATE TABLE
# =====================================================

def create_prediction_table():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT,

            smiles TEXT,

            model_used TEXT,

            prediction_k REAL,

            prediction_c REAL,

            status TEXT,

            created_at TEXT
        )
        """
    )

    conn.commit()

    conn.close()


# =====================================================
# INSERT LOG
# =====================================================

def log_prediction(
    username,
    smiles,
    model_used,
    prediction_k,
    prediction_c,
    status="Success"
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_logs (

            username,
            smiles,
            model_used,
            prediction_k,
            prediction_c,
            status,
            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            smiles,
            model_used,
            prediction_k,
            prediction_c,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()

    conn.close()


# =====================================================
# LOAD LOGS
# =====================================================

def load_prediction_logs():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT

            id,
            username,
            smiles,
            model_used,
            prediction_k,
            prediction_c,
            status,
            created_at

        FROM prediction_logs

        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


# =====================================================
# CLEAR ALL LOGS
# =====================================================

def clear_prediction_logs():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM prediction_logs"
    )

    conn.commit()

    conn.close()


# =====================================================
# DELETE SINGLE ROW
# =====================================================

def delete_prediction_row(row_id):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM prediction_logs
        WHERE id = ?
        """,
        (row_id,)
    )

    conn.commit()

    conn.close()