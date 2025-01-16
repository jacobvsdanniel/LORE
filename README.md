# LORE

**[A Literature Semantics Framework with LLMs](https://doi.org/10.1101/2024.08.10.24311801) to build knowledge graphs, embeddings, and association predictors**

**[The LORE PMKB-CV Dataset](https://doi.org/10.5281/zenodo.14607638) contains PubMed disease-gene knowledge graphs, embeddings, and predicted pathogenicity scores**

Source code authors:
- Li Peng-Hsuan (李朋軒) (jacobvsdanniel [at] gmail.com) @ Taiwan AI Labs

Researchers:
- Li Peng-Hsuan (李朋軒) (jacobvsdanniel [at] gmail.com) @ Taiwan AI Labs
- Sun Yih-Yun (孫懿筠) (jessie.yy.sun [at] gmail.com) @ Taiwan AI Labs
- Juan Hsueh-Fen (阮雪芬) (yukijuan [at] gmail.com) @ National Taiwan University
- Chen Chien-Yu (陳倩瑜) (chienyuchen [at] g.ntu.edu.tw) @ National Taiwan University
- Tsai Huai-Kuang (蔡懷寬) (hktsai616 [at] gmail.com) @ Academia Sinica
- Huang Jia-Hsin (黃佳欣) (jiahsin.huang [at] ailabs.tw) @ Taiwan AI Labs

## The LORE Literature Semantics Framework

LORE consists of four core modules:

- **LLM-ORE**
  - Curates an entity-entity relations **Knowledge Graph** from literature articles using LLM-based open relation extraction

- **LLM-EMB**
  - Creates an entity-entity **Semantic Embedding** using LLMs reading the knowledge graph

- **ML-Ranker**
  - Builds an entity-entity **Association Score Predictor** using the embedding and sparse positive labels

- **Key-Semantics**
  - Annotates a controlled list of **Semantic Tags** and add to knowledge graph relations

For more details, see our paper:

- Peng-Hsuan Li, Yih-Yun Sun, Hsueh-Fen Juan, Chien-Yu Chen, Huai-Kuang Tsai, and Jia-Hsin Huang. 2024. [LORE: A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale.](https://doi.org/10.1101/2024.08.10.24311801)

## The LORE PMKB-CV Dataset

We have run LORE on all **4M PubMed article abstracts** that have Disease-Gene or Disease-Variant co-occurrences and created:

- Knowledge graph (LLM-ORE)
  - **70M relations** between **8k D**iseases (MeSH) and **18k G**enes (NCBI, human protein coding) curated by LLMs reading PubMed
  - Data format: (D_id, G_id, PMID, relation) csv file

- Semantic embedding (LLM-EMB)
  - **2.5M DG vectors** created by LLMs reading the knowledge graph
  - Data format: (D_id, G_id, vector) pkl file

- DG pathogenicity scores (ML-Ranker)
  - **3.1M DG scores** predicted by pretrained models
  - Features, training annotations, pretrained models are also provided

- Curated key semantics taxonomy
  - A manually curated taxonomy of **105 semantic tags** about DG pathogenicity in the knowledge graph
  - Use the github LORE Key-Semantics module to use the taxonomy as tags and add them to the knowledge graph

The dataset is publicly available:
- Li, P.-H. (2025). LORE PMKB-CV [Data set]. Taiwan AI Labs. https://doi.org/10.5281/zenodo.14607639

## Running LORE for Custom Literature Articles and Entities

### Installation

```
git clone https://github.com/jacobvsdanniel/LORE.git
cd LORE
pip install -r requirements.txt
```

### LLM-ORE

- Using gpt-4o-mini hosted by OpenAI servers

```
python LORE.py --config_file examples/config_LLM-ORE_gpt-4o-mini.json
```

- Using meta-llama/Llama-3.1-8B-Instruct hosted by Deep Infra servers

```
python LORE.py --config_file examples/config_LLM-ORE_llama-8b.json
```

- Customize config_LLM-ORE.json for:

  - server: OpenAI, Deep Infra, your own local vLLM server, any server that supports OpenAI client call
  - model: tell the server your desired LLM
  - prompt: you can provide your customized instructions
  - file path to input articles and entities (example jsonl files provided)
  - file path to output knowledge graphs (example csv files provided)

### LLM-EMB

- Using text-embedding-3-large hosted by OpenAI servers

```
python LORE.py --config_file examples/config_LLM-EMB.json
```

- Customize config_LLM-EMB.json for

  - model: tell the server your desired LLM
  - dimension: your desired embedding dimension
  - file path to input knowledge graphs (example csv file provided)
  - file path to output embedding (example pkl file provided, content: a list of (P_id, G_id, numpy_ndarray) )

### ML-Ranker

All the config, labels, and features (for 5k samples) in the following scenarios are included in ./examples/ML-Ranker

This is enough for preparing your custom dataset with correct data formats and train your own models.

However, to run the following scenarios and reproduce expected results, please download the [LORE PMKB-CV](https://doi.org/10.5281/zenodo.14607639) dataset and uncompress to ./PMKB-CV

- Scenario: k-fold cross validation

  - Saves predicted association scores
  - Evaluates performance

```
python LORE.py --config_file PMKB-CV/2025/ML-Ranker/setting_k-fold/config.json
# expected stdout:
# [DG label] 4,311 DGs: 3,175 unique Ds, 2,416 unique Gs
# [DG feature] 652,701 DGs: 2,097 unique Ds, 17,750 unique Gs
# [DG embedding] 541,474 DGs: 2,065 unique Ds, 17,602 unique Gs
# 5 fold cross-validation, #D-per-fold: [419, 419, 419, 420, 420]
# MAP=81.3% proportion_of_known_positive_DGs_predicted=94.8%
```

- Scenario: leave-one-out cross validation

  - Saves predicted association scores
  - Evaluates performance

```
python LORE.py --config_file PMKB-CV/2025/ML-Ranker/setting_leave-one-out/config.json
# expected stdout:
[DG label] 4,311 DGs: 3,175 unique Ds, 2,416 unique Gs
[DG feature] 652,701 DGs: 2,097 unique Ds, 17,750 unique Gs
[DG embedding] 541,474 DGs: 2,065 unique Ds, 17,602 unique Gs
2,097 fold cross-validation, #D-per-fold: 1
MAP=81.6% proportion_of_known_positive_DGs_predicted=94.8%
```

- Scenario: training a predictor

  - Saves the trained predictor model

```
python LORE.py --config_file PMKB-CV/2025/ML-Ranker/setting_train-test/config_train.json
# expected stdout:
# [DG label] 4,311 DGs: 3,175 unique Ds, 2,416 unique Gs
# [DG feature] 652,701 DGs: 2,097 unique Ds, 17,750 unique Gs
# [DG embedding] 541,474 DGs: 2,065 unique Ds, 17,602 unique Gs
```

- Scenario: testing a predictor

  - Saves predicted association scores
  - (optional) Evaluates performance if label file is provided

```
python LORE.py --config_file PMKB-CV/2025/ML-Ranker/setting_train-test/config_test.json
# expected stdout:
# [DG feature] 3,128,402 DGs: 8,894 unique Ds, 18,393 unique Gs
# [DG embedding] 2,556,839 DGs: 8,561 unique Ds, 18,343 unique Gs
# [DG label] 4,311 DGs: 3,175 unique Ds, 2,416 unique Gs
# MAP=88.3% proportion_of_known_positive_DGs_predicted=94.8%
```

### Key-Semantics

- Step 1: Preprocess
  
  - Extracts lemmas for the input knowledge graph
  - Creates a list of candidate high coverage, high precision lemmas to be used as relation tags
  - Samples a set of relations for each candidate tags to aid manual inspection
    
```
python LORE.py --config_file examples/config_Key-Semantics_extraction.json
```

- Step 2: Curation

Option A - your own list
```
# Inspect the <semantics_candidate_file> created in step 1
# Create your curated list of key semantics, see ./examples/Key-Semantics_semantics.csv
```

Option B - PMKB-CV taxonomy
```
# Download the LORE PMKB-CV (https://doi.org/10.5281/zenodo.14607639) dataset and uncompress to ./PMKB-CV
# The ./PMKB-CV/key_semantics_taxonomy/taxonomy will be used in the next step
```

- Step 3: tagging

For option A - your own list
```
python LORE.py --config_file examples/config_Key-Semantics_tagging_list.json
```

For option B - PMKB-CV taxonomy
```
python LORE.py --config_file examples/config_Key-Semantics_tagging_taxonomy.json
```

See ./examples/Key-Semantics_knowledge_graph.csv for an example tagged knowledge graph.

## Citation

If this work is helpful, please kindly cite as:

- Peng-Hsuan Li, Yih-Yun Sun, Hsueh-Fen Juan, Chien-Yu Chen, Huai-Kuang Tsai, and Jia-Hsin Huang. 2024. [LORE: A Literature Semantics Framework for Evidenced Disease-Gene Pathogenicity Prediction at Scale.](https://doi.org/10.1101/2024.08.10.24311801)
- Li, P.-H. (2025). LORE PMKB-CV [Data set]. Taiwan AI Labs. https://doi.org/10.5281/zenodo.14607639
