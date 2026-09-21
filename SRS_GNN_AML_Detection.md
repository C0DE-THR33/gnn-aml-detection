# Software Requirements Specification

## Explainable Anti-Money Laundering Detection using Heterophily-Aware Graph Neural Networks

| | |
|---|---|
| **Course** | AI in Finance (CSF 3127) |
| **Institution** | MIT Manipal |
| **Author** | Pratham (Reg. No. 240958098) |
| **Document Version** | 1.0 |
| **Date** | September 2026 |
| **Status** | Draft |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements for a system that detects money-laundering transactions in a financial transaction graph using a heterophily-aware Graph Neural Network (GNN), and generates human-interpretable explanations for each flagged transaction using GNNExplainer. It is intended to guide implementation, evaluation, and grading of the CSF 3127 course project, and to serve as a reference for scope control.

### 1.2 Scope
The system ingests a labeled transaction dataset, constructs a transaction graph, trains a GNN-based edge classifier to flag illicit (money-laundering) transactions, generates a local explanatory subgraph for each flagged transaction via GNNExplainer, and evaluates those explanations against ground-truth laundering typology labels.

**In scope:**
- Graph construction from the IBM AML (HI-Small) dataset
- Edge-level (transaction-level) binary classification (licit / illicit)
- A heterophily-aware GNN architecture
- Instance-level explanation generation via GNNExplainer
- Per-typology evaluation of both classification performance and explanation fidelity

**Dataset selection (locked in):** HI-Small is used, not LI-Small, Medium, or Large. HI-Small is the only variant small enough to train and run instance-level explanations on with commodity hardware (~515K accounts / ~5.08M transactions over 10 days, vs. ~32M transactions for Medium and ~175M–180M for Large). Within the Small tier, HI is used over LI because HI's illicit ratio is roughly double LI's — LI-Small's already-thin per-typology counts would shrink further, undermining the per-typology fidelity evaluation this project is built around.

**Out of scope (explicitly deferred to future work):**
- Temporal / sequential modeling of transaction streams
- Node-level or full-subgraph (ring) detection as a standalone task
- Multi-dataset generalization (Elliptic, PaySim, etc.)
- Comparative benchmarking against other explainers (PGExplainer, SubgraphX)
- Real-time / streaming inference
- Deployment as a production service

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|---|---|
| AML | Anti-Money Laundering |
| GNN | Graph Neural Network |
| GAT | Graph Attention Network |
| GCN | Graph Convolutional Network |
| GraphSAGE | Graph Sample and Aggregate |
| Heterophily | A tendency for connected nodes to have *dissimilar* labels/features, as opposed to homophily |
| GNNExplainer | A model-agnostic method that identifies a compact subgraph and feature subset that most influence a GNN's prediction for a given instance |
| Typology | A known structural pattern of money laundering (e.g., fan-out, cycle) |
| HI-Small | The "High Illicit ratio, Small size" variant of the IBM AML synthetic dataset |
| Fidelity | The degree to which an explanation subgraph reflects the actual structure the model (or ground truth) relied on |

### 1.4 References
- IBM Transactions for Anti-Money Laundering (AML) dataset documentation
- Ying et al., *GNNExplainer: Generating Explanations for Graph Neural Networks*
- PyTorch Geometric (PyG) documentation
- Course materials, AI in Finance (CSF 3127)

### 1.5 Overview
Section 2 describes the system at a high level. Section 3 lists specific functional, data, and non-functional requirements. Section 4 describes the intended architecture. Section 5 defines the evaluation plan and acceptance criteria. Section 6 contains supporting appendices.

---

## 2. Overall Description

### 2.1 Product Perspective
This is a standalone academic research artifact, not an integration into an existing system. It consists of a data pipeline, a training pipeline, and an explanation/evaluation pipeline, run offline over a static dataset. There is no live production interface.

### 2.2 Product Functions (Summary)
1. Load and preprocess the IBM AML HI-Small dataset into a graph representation.
2. Train a heterophily-aware GNN to classify transactions (edges) as licit or illicit.
3. Evaluate classification performance overall and broken down by known typology.
4. For a sample of correctly-flagged illicit transactions, generate a GNNExplainer explanation subgraph.
5. Score explanation fidelity per typology by comparing the explanation subgraph's structure to the known structural signature of that typology.
6. Produce a final report summarizing model performance and explanation quality.

### 2.3 User Characteristics
The primary user is the project author/evaluator (course instructor). No end-user-facing UI is required; outputs are notebooks, scripts, plots, and a written report.

### 2.4 Constraints
- Must be implementable within a single academic semester by one student.
- Must run on commodity hardware (single GPU or CPU, no distributed training).
- Must use PyTorch Geometric (PyG) as the primary GNN library, for compatibility with GNNExplainer's built-in implementation.
- Dataset is fixed to HI-Small (over LI-Small, HI-Medium/Large, LI-Medium/Large) to keep training and explanation generation tractable while preserving enough per-typology illicit examples for evaluation.

### 2.5 Assumptions and Dependencies
- The IBM AML HI-Small dataset is accessible and its typology labels are usable for evaluation (not just training).
- GNNExplainer's default implementation in PyG is sufficient; no custom explainer implementation is required.
- Class imbalance (illicit transactions are a small minority) will require explicit handling (e.g., weighted loss, oversampling) but does not change the overall system design.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | The system shall load the IBM AML HI-Small dataset and construct a directed transaction graph (accounts as nodes, transactions as edges). | Must |
| FR-2 | The system shall attach available node features (e.g., account/bank metadata) and edge features (e.g., amount, currency, timestamp, payment format) to the graph. | Must |
| FR-3 | The system shall split the dataset into train/validation/test sets in a way that avoids label leakage across the graph. | Must |
| FR-4 | The system shall implement a heterophily-aware GNN architecture (e.g., separate self- and neighbor-embedding aggregation) for edge classification. | Must |
| FR-5 | The system shall implement at least one homophily-assuming baseline (e.g., standard GraphSAGE or GAT) for comparison against the heterophily-aware model. | Should |
| FR-6 | The system shall train the classifier with a loss function appropriate for class imbalance (e.g., weighted cross-entropy or focal loss). | Must |
| FR-7 | The system shall report standard classification metrics (precision, recall, F1, AUPRC) on the test set, both overall and broken down by ground-truth typology. | Must |
| FR-8 | The system shall select a sample of correctly-classified true-positive (illicit) transactions for explanation. | Must |
| FR-9 | The system shall generate a GNNExplainer explanation (important subgraph + feature mask) for each selected transaction. | Must |
| FR-10 | The system shall define, for each supported typology, a reference structural signature (e.g., fan-out = one source node with many distinct destination edges within a short window) against which explanations can be compared. | Must |
| FR-11 | The system shall compute an explanation fidelity score per typology, quantifying how well GNNExplainer's output subgraph matches the expected structural signature. | Must |
| FR-12 | The system shall produce visualizations of representative explanation subgraphs for each typology. | Should |
| FR-13 | The system shall produce a final written report summarizing dataset statistics, model performance, and per-typology explanation fidelity. | Must |

### 3.2 Data Requirements
- **Source:** IBM Transactions for Anti-Money Laundering, HI-Small variant.
- **Format:** Tabular transaction records convertible into a graph (nodes = accounts, edges = transactions).
- **Labels required:** Binary illicit/licit label per transaction; typology label per illicit transaction, drawn from the 8 typologies in `HI-Small_Patterns.txt` (fan-out, fan-in, cycle, gather-scatter, scatter-gather, stack, bipartite, random). A residual "not classified" bucket exists in the transaction-level labels and is excluded from per-typology evaluation rather than forced into one of the 8 categories.
- **Preprocessing:** Deduplication, timestamp normalization, currency normalization (if cross-currency transactions are present), train/val/test partitioning that respects graph connectivity.

### 3.3 Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Reproducibility:** All experiments shall be runnable end-to-end from a documented script/notebook with fixed random seeds. |
| NFR-2 | **Performance:** Full training on HI-Small shall complete in under a few hours on a single consumer GPU (or be subsampled if this is not achievable). |
| NFR-3 | **Interpretability:** Explanation outputs shall be renderable as a visual subgraph, not just raw tensors. |
| NFR-4 | **Maintainability:** Code shall be organized into clearly separated modules (data loading, model, training, explanation, evaluation). |
| NFR-5 | **Documentation:** Each module shall include enough inline documentation for the report/evaluator to trace results back to code. |

### 3.4 External Interface Requirements
- **Input interface:** Local dataset files (CSV/Parquet as distributed by IBM AML).
- **Output interface:** Trained model checkpoint, metrics tables (CSV/Markdown), explanation subgraph visualizations (PNG/SVG), final report (Markdown/PDF).
- No network-facing API or UI is required.

---

## 4. System Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Data Ingestion  │ --> │  Graph Construc-  │ --> │  Heterophily-Aware    │
│  (IBM AML HI-S)  │     │  tion & Features   │     │  GNN Training         │
└─────────────────┘     └──────────────────┘     └───────────┬───────────┘
                                                               │
                                                               v
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Per-Typology    │ <-- │  Explanation      │ <-- │  Edge Classification  │
│  Fidelity Eval   │     │  (GNNExplainer)    │     │  (Test Predictions)   │
└─────────────────┘     └──────────────────┘     └───────────────────────┘
        │
        v
┌─────────────────┐
│  Final Report    │
└─────────────────┘
```

**Pipeline stages:**
1. **Data Ingestion & Graph Construction** — load HI-Small, build a heterogeneous or homogeneous transaction graph with node/edge features.
2. **Model Training** — train the heterophily-aware GNN (and baseline, if included) as an edge classifier.
3. **Classification Evaluation** — compute metrics overall and per typology.
4. **Explanation Generation** — run GNNExplainer on a sample of true-positive predictions.
5. **Explanation Evaluation** — score fidelity per typology against known structural signatures.
6. **Reporting** — consolidate all results into the final report.

---

## 5. Evaluation Plan and Acceptance Criteria

| Deliverable | Acceptance Criterion |
|---|---|
| Graph construction | Graph correctly represents accounts/transactions with no leakage between train/test splits |
| Classifier | Heterophily-aware model achieves measurably better recall/F1 on illicit class than the homophily baseline |
| Per-typology metrics | Metrics reported separately for each of the 8 known typologies, not just in aggregate |
| Explanations | GNNExplainer runs successfully on all sampled true positives without failure |
| Fidelity scoring | A defined, reproducible scoring method exists for comparing explanation subgraphs to typology signatures |
| Report | Report clearly states which typologies are well-explained vs. poorly-explained, with visual examples |

---

## 6. Appendix

### 6.1 AML Typologies Considered

| Typology | Structural Signature (informal) |
|---|---|
| Fan-out | One source account sends to many distinct destination accounts in a short window |
| Fan-in | Many source accounts send to one destination account in a short window |
| Cycle | Funds return to (approximately) the originating account after a chain of transfers |
| Gather-scatter | Funds are gathered from many sources into one account, then scattered to many destinations |
| Scatter-gather | Funds are scattered from one source to many intermediaries, then gathered into one account |
| Stack | Sequential chain of transfers through a series of intermediary accounts |
| Bipartite | Transactions concentrated between two distinct account groups, with little flow outside the pair of groups |
| Random | Transactions that don't fit a structured typology but are still labeled illicit (a weaker, catch-all signal) |

Note: `HI-Small_Patterns.txt` also contains a "not classified" residual bucket of labeled-illicit transactions that don't map cleanly to any of the 8 typologies above. These are excluded from per-typology fidelity scoring (see FR-11) rather than assigned to a typology they don't structurally match.

### 6.2 Explicit Non-Goals (for scope defense during review)
- No claim of production-grade AML compliance capability is made.
- No claim of generalization beyond the HI-Small synthetic dataset is made.
- Explanation fidelity is evaluated against known synthetic typology structure, not against real investigator judgment.

### 6.3 Risk Register

| Risk | Mitigation |
|---|---|
| Severe class imbalance hurts recall | Weighted loss / oversampling (FR-6) |
| GNNExplainer output not meaningfully different from random subgraph on some typologies | Report this honestly as a limitation rather than omitting weak results |
| Training time too long on full HI-Small | Subsample transactions while preserving typology proportions |
| Heterophily-aware model shows no improvement over baseline | Still reportable as a valid negative result if baseline (FR-5) is included |
