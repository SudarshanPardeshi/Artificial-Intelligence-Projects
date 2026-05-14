# 🧪 Hybrid GNN AI Cheminformatics Platform

An advanced AI-powered cheminformatics research platform for molecular melting point prediction, molecular similarity analysis, uncertainty estimation, OOD detection, scaffold analysis, and interactive chemical space visualization.

---

# 🚀 Features

## 🔥 AI-Based Melting Point Prediction
- RDKit descriptor-based ML model
- Hybrid GAT AI model
- Ensemble AI prediction
- Confidence estimation
- Uncertainty estimation

---

## 🔍 Molecule Explorer
- Search molecules using:
  - IUPAC names
  - SMILES
- Interactive dataset exploration
- Molecule copy support

---

## 🧬 Molecular Similarity Search
- Top 10 similar molecules
- Morgan fingerprint similarity
- Tanimoto similarity scoring

---

## 🖼 Molecular Visualization
- 2D molecular structure rendering
- PNG export
- Interactive visualization support

---

## 📊 Advanced Chemical Space Visualization

### PCA Visualization
- Global molecular distribution
- Dataset diversity analysis
- Outlier visualization

### t-SNE Visualization
- Local similarity clustering
- Hidden molecular clusters

### UMAP Visualization
- Advanced chemical space mapping
- Scaffold-family clustering
- Research-grade visualization

### Interactive Plotly UMAP
- Zoom & pan
- Hover molecule details
- Interactive exploration
- AI overlay visualization

---

# 🤖 AI Overlay Visualization

Interactive chemical space coloring by:
- Predicted melting point
- Confidence percentage
- OOD status
- Murcko scaffold
- Molecular groups

---

# 🧱 Scaffold Analysis
- Murcko scaffold extraction
- Scaffold frequency analysis
- Core structure exploration
- Scaffold clustering

---

# ⚠️ OOD (Out-of-Distribution) Detection
Detect whether a molecule is unlike training chemistry.

Features:
- Similarity-based OOD detection
- Reliability estimation
- Prediction trust analysis
- Nearest molecule matching

---

# 📄 AI PDF Report Generator
Generate professional prediction reports containing:
- Molecular information
- Predictions
- Confidence estimation
- SHAP explainability
- Similar molecules
- OOD analysis
- Model comparison

---

# 📦 Batch CSV Prediction
- Batch prediction using CSV upload
- Ensemble prediction
- Confidence estimation
- Batch PDF summary generation

---

# 📈 Dashboard Analytics
Includes:
- Total predictions
- Confidence distribution
- Prediction success rate
- Average melting point
- Usage analytics

---

# 🧠 Technologies Used

## AI / Machine Learning
- LightGBM
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn

## Visualization
- Plotly
- Matplotlib
- UMAP
- t-SNE
- PCA

## Web Framework
- Streamlit

## Reporting
- ReportLab

---

# 📂 Project Structure

```text
Hybrid_GNN_AI_Cheminformatics/
│
├── streamlit_app.py
├── requirements.txt
├── config.yaml
├── all_smiles_with_names.csv
│
├── saved_models/
│   ├── rdkit_model.pkl
│   ├── hybrid_gat_model.pt
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── reports/
├── outputs/
├── figures/
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

### RDKit (Windows Recommended)

```bash
conda install -c conda-forge rdkit
```

---

# ▶️ Run Application

```bash
streamlit run streamlit_app.py
```

---

# 📊 Research-Grade Features

This platform integrates:
- Explainable AI
- Uncertainty-aware prediction
- OOD detection
- Scaffold analysis
- Interactive chemical space exploration

making it suitable for:
- AI research
- cheminformatics research
- academic projects
- publication-quality workflows
- portfolio projects

---

# 🧪 Example Capabilities

✅ Predict melting point from SMILES  
✅ Search molecules by IUPAC name  
✅ Detect unreliable predictions  
✅ Visualize molecular clusters  
✅ Explore scaffold families  
✅ Generate AI reports  
✅ Perform batch predictions  

---

# 📌 Future Improvements

- Transformer-based molecular models
- Attention visualization
- Molecular docking integration
- Drug-likeness prediction
- Live PubChem API integration
- Cloud deployment
- GPU optimization

---

# 👨‍💻 Author

Sudarshan Pardeshi

AI + Cheminformatics + Machine Learning Research

---

# 📜 License

This project is intended for:
- educational use
- research purposes
- AI experimentation

---