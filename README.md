# LORE
**A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale**

Source code authors:
- Li Peng-Hsuan (李朋軒) @ ailabs.tw (jacobvsdanniel [at] gmail.com)

## Introduction

This repo hosts the source codes for LORE (LLM-based Open Relation Extraction and Embedding). We applied LORE to PubMed abstracts for large-scale understanding of disease-gene relationships and created the PMKB-CV knowledge graph. PMKB-CV contains 2K diseases, 600K disease-gene pairs, 11M disease-gene relations, embeddings, and predicted pathogenicity scores. This resource covers 200x more disease-gene pairs than ClinVar, and the predicted pathogenicity scores achieve an 80% Mean Average Precision (MAP) in ranking pathogenic genes for diseases.

**For more details, see our paper:**

Peng-Hsuan Li, Yih-Yun Sun, Hsueh-Fen Juan, Chien-Yu Chen, Huai-Kuang Tsai, and Jia-Hsin Huang. 2024. LORE: A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale.

**The PMKB-CV knowledge graph is publicly available at:**

[LORE-PMKB-CV](https://drive.google.com/file/d/1rGgZmUOU0XIQtV3mtYsMU-4t2lJQNOfo) © 2024 by [Taiwan AI Labs](https://ailabs.tw), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0). <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg">
