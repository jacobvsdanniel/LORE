# LORE
**A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale**

Source code authors:
- Li Peng-Hsuan (李朋軒) @ ailabs.tw (jacobvsdanniel [at] gmail.com)

## Introduction

This repo hosts the source codes for LORE (LLM-based Open Relation Extraction and Embedding). We applied LORE to PubMed abstracts for large-scale understanding of disease-gene relationships and created the PMKB-CV knowledge graph. PMKB-CV contains 2K diseases, 600K disease-gene pairs, 11M disease-gene relations, embeddings, and predicted pathogenicity scores. This resource covers 200x more disease-gene pairs than ClinVar, and the predicted pathogenicity scores achieve an 80% Mean Average Precision (MAP) in ranking pathogenic genes for diseases. The PMKB-CV knowledge graph is available at: [PMKB-CV](https://drive.google.com/file/d/1rGgZmUOU0XIQtV3mtYsMU-4t2lJQNOfo).

For more details, see our paper:

*Peng-Hsuan Li, Yih-Yun Sun, Hsueh-Fen Juan, Chien-Yu Chen, Huai-Kuang Tsai, and Jia-Hsin Huang. 2024. **A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale.***
