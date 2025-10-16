# 🧠 AI-SupplyShield 🔒  
### AI-Based Real-Time Threat Analysis for Software Supply Chains  

---

## 📘 Overview
**AI-SupplyShield** is an intelligent framework designed to **detect, analyze, and respond to software supply chain attacks in real time**.  
It leverages **Machine Learning (ML)** and **Deep Learning (DL)** to profile the behavior of open-source packages and identify anomalies such as **Trojanized dependencies**, **typosquatting**, and **build-pipeline compromises**.

The project is part of a **Computer & Network Security** course focusing on AI-driven cybersecurity innovations.

---

## 🎯 Objectives
- Collect and preprocess open-source software telemetry and behavioral data.  
- Profile normal vs. malicious package behavior using ML/DL models.  
- Detect software supply-chain attacks such as:
  - Trojanized open-source dependencies  
  - Dependency confusion / typosquatting  
  - Build-pipeline injection attacks  
- Design and evaluate a **real-time anomaly detection pipeline**.  
- Implement **automated response simulation** (alert generation or quarantine).  

---

## 💡 Motivation
Modern software ecosystems depend heavily on third-party libraries and CI/CD automation.  
Attackers exploit this trust chain to inject malicious code during development or distribution — often **bypassing traditional firewalls and endpoint security tools**.

AI-SupplyShield aims to provide an **adaptive and intelligent defense** that continuously learns from behavior patterns and detects threats **before exploitation spreads through the supply chain**.

---

## 🧠 System Architecture

1. **Data Ingestion Layer** – Collects and normalizes logs (system calls, network traces, install behavior).  
2. **Feature Extraction Layer** – Derives behavioral features from package execution (I/O patterns, DNS queries, entropy, etc.).  
3. **ML/DL Detection Engine** – Trains models (Autoencoder / LSTM / Isolation Forest) for anomaly classification.  
4. **Real-Time Monitor** – Continuously analyzes new package installations or updates.  
5. **Automated Response Module** – Flags, isolates, or logs suspicious activity.  

---

## 📊 Datasets Used

| Dataset | Description | Source |
|----------|--------------|---------|
| **QUT-DV25 Dataset** | Dynamic analysis of 14K PyPI packages (7K malicious) with system & network logs. | [ArXiv:2505.13804](https://arxiv.org/abs/2505.13804) |
| **OSTrack Dataset** | Runtime and static features for 9K open-source packages labeled by behavior. | [ArXiv:2411.14829](https://arxiv.org/abs/2411.14829) |
| **Backstabber’s Knife Collection** | Real-world malicious OSS packages (npm, PyPI, RubyGems). | [ArXiv:2005.09535](https://arxiv.org/abs/2005.09535) |

> Optionally supplemented by:  
> [Atlantic Council – Breaking Trust Dataset](https://www.atlanticcouncil.org/commentary/trackers-and-data-visualizations/breaking-trust-the-dataset/)  
> and [GitHub Supply-Chain Incident Repo](https://github.com/tstromberg/supplychain-attack-data)

---

## 🧩 Implementation Flow

```mermaid
graph TD
A[Data Collection] --> B[Preprocessing & Feature Engineering]
B --> C[Model Training]
C --> D[Anomaly Detection]
D --> E[Automated Response Simulation]
E --> F[Evaluation & Reporting]
