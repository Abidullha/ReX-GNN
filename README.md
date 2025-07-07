# 🧠 ReX-GNN

**ReX-GNN** (Real-time Explainable Graph Neural Network) is a modular, interpretable spatio-temporal model for traffic forecasting.  
It uses GCN + Attention + GRU to capture both spatial and temporal dynamics in road networks like **METR-LA** and **PEMS-BAY**.

---

## 📦 Project Structure

ReX-GNN/
├── data/ # METR-LA, PEMS-BAY data (not versioned)
├── models/ # ReX-GNN model (GCN + Attn + GRU)
├── notebooks/ # Colab or Jupyter notebooks
├── utils/ # Preprocessing + evaluation helpers
├── main.py # Training script
├── requirements.txt # Project dependencies
└── README.md # This file

---

## 🚀 Features

- ✅ Graph Convolutional Network (GCN)
- ✅ Spatial Attention for interpretability
- ✅ GRU for temporal modeling
- ✅ Temporal Attention over past time steps
- ✅ Modular components: replace/ablate any part
- ✅ Human-readable insights from attention weights

---

## 🗂️ Datasets

- **METR-LA**: Traffic speed from 207 sensors in Los Angeles  
- **PEMS-BAY**: Traffic speed from 325 sensors in the Bay Area

Links & usage instructions in `data/README.md`.

---

## 🔧 Setup

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
## 🧪 Run Training

Run the main training loop:

```bash
python main.py

## Citation
@inprogress{rexgnn2025,
  title={ReX-GNN: Modular and Interpretable GNN for Traffic Forecasting},
  author={Your Name},
  year={2025},
  note={In preparation}
}
