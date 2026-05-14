\# 🧪 Melting Point Prediction AI Platform



An advanced AI-powered cheminformatics platform for predicting molecular melting points using:



\- RDKit molecular descriptors

\- LightGBM machine learning

\- Hybrid Graph Attention Network (GAT)

\- Explainable AI (SHAP)

\- Streamlit deployment UI

\- FastAPI backend support



\---



\# 🚀 Features



\## ✅ AI Models



\### 1. RDKit + LightGBM

Traditional descriptor-based machine learning model using:

\- RDKit descriptors

\- engineered molecular features

\- LightGBM regression



\### 2. Hybrid Descriptor + GAT

Advanced hybrid AI architecture combining:

\- RDKit descriptors

\- Graph Attention Network embeddings

\- RandomForest hybrid regression



\---



\# 🧬 Explainable AI



\## RDKit LightGBM SHAP

\- SHAP feature explanations

\- SHAP contribution plots

\- descriptor-level interpretability



\## Hybrid GAT SHAP

\- Hybrid embedding SHAP analysis

\- GAT embedding importance

\- feature contribution analysis



\---



\# 🧪 Molecular Features



The platform displays:

\- Molecular structure visualization

\- Molecular formula

\- Molecular weight

\- LogP

\- TPSA

\- H-Bond Donors

\- H-Bond Acceptors

\- Rotatable Bonds

\- Ring Count



\---



\# 📊 Prediction Modes



\## Single Molecule Prediction

Predict melting point for:

\- dropdown SMILES selection

\- custom user SMILES



\## Batch CSV Prediction

Upload CSV containing:

```text

SMILES



and generate:



batch predictions

downloadable CSV

Full Dataset Prediction



Run predictions for:



full cleaned molecular dataset

complete AI inference workflow



🌡️ Prediction Outputs



Predictions shown in:



Kelvin (K)

Celsius (°C)





🧠 Models Used

Model				Purpose

LightGBM			Descriptor-based regression

GAT (PyTorch Geometric)		Molecular graph embedding

RandomForest			Hybrid regression

SHAP				Explainable AI





🛠️ Technologies Used

Python

RDKit

PyTorch

PyTorch Geometric

LightGBM

Scikit-learn

SHAP

Streamlit

FastAPI

Pandas

NumPy

Matplotlib



📁 Project Structure

03\_GNN\_Melting\_Point\_Deployment/

│

├── streamlit\_app.py

├── api.py

├── inference.py

├── hybrid\_inference.py

├── gnn\_utils.py

├── gat\_model.py

├── rdkit\_utils.py

│

├── rdkit\_lightgbm\_model.pkl

├── hybrid\_gat\_model.pkl

├── gat\_model.pth

├── hybrid\_feature\_names.pkl

├── feature\_columns.pkl

│

├── all\_smiles\_clean.csv

├── requirements.txt

├── README.md





▶️ Run Streamlit App

streamlit run streamlit\_app.py





▶️ Run FastAPI Backend

uvicorn api:app --reload



Open:



http://127.0.0.1:8000







📦 Install Requirements

pip install -r requirements.txt





🧪 Example SMILES

CCO

CCN

C

O=C=O

c1ccccc1





📈 Explainability Outputs

RDKit SHAP

top descriptor contributions

SHAP bar charts



Hybrid GAT SHAP

embedding SHAP analysis

hybrid feature importance

GAT embedding contribution analysis





🔥 Advanced AI Features

Multi-model AI platform

Hybrid graph learning

Explainable AI

Molecular graph embeddings

Descriptor engineering

Batch inference system

Production deployment architecture





🌍 Deployment Targets

Local deployment

Docker deployment

Streamlit Cloud deployment

FastAPI REST API deployment

GitHub integration





📚 Research Areas

Cheminformatics

Molecular AI

Thermophysical property prediction

Graph Neural Networks

Explainable AI

Scientific machine learning


# 🚀 Advanced Platform Features

## 🔐 User Authentication
- Secure login system using `streamlit-authenticator`
- Session-based authentication
- Protected AI prediction dashboard
- Logout functionality included

---

## 🗄️ SQLite Prediction Logging
- Automatic prediction logging system
- Stores:
  - Username
  - SMILES input
  - Model used
  - Predicted melting point
  - Prediction status
  - Timestamp

---

## 📊 Prediction History Dashboard
- Dedicated prediction history tab
- Real-time history tracking
- Interactive prediction monitoring
- Download history as CSV

---

## 🗑️ Delete Selected Predictions
- Delete individual prediction records
- Row-level prediction management
- Interactive deletion system

---

## 🧹 Clear Entire Prediction History
- Remove all prediction logs instantly
- Database reset functionality
- One-click cleanup system

---

# 🐳 Docker Deployment
- Fully containerized AI platform
- Portable deployment architecture
- Production-ready Docker setup
- Streamlit application deployed inside Docker container

### Docker Commands

```bash
docker build -t hybrid-gnn-ai-cheminformatics-platform .
docker run -p 8501:8501 hybrid-gnn-ai-cheminformatics-platform




☁️ Streamlit Cloud Deployment
Public cloud-hosted AI application
Accessible from anywhere
Live online prediction platform
Integrated GitHub deployment pipeline


🏗️ Production Architecture
User
↓
Authentication Layer
↓
Streamlit Frontend
↓
Hybrid AI Prediction Engine
↓
RDKit + LightGBM + Hybrid GAT Models
↓
SQLite Prediction Logging Database


💡 Enterprise-Level Features

✅ AI Explainability (SHAP)
✅ Molecular Visualization
✅ Multi-Model Prediction
✅ Batch CSV Prediction
✅ Full Dataset Prediction
✅ User Authentication
✅ Database Logging
✅ Docker Deployment
✅ Cloud Deployment
✅ Prediction History Management


👨‍💻 Author



Sudarshan Pardeshi



AI-Based Thermophysical Property Prediction System

Melting Point Estimation using Hybrid Descriptor + Graph Neural Networks





⭐ Future Improvements

GPU inference optimization

Authentication system

Database logging

Online REST API hosting

Cloud inference scaling

Full production monitoring





📜 License



Educational and research use.





Save using:



```text

CTRL + S



