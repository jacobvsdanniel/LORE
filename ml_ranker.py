import os
import json
import pickle
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
feature_list = [
    "paper", "sentence",
    "cre_Cause-associated", "cre_Appositive", "cre_In-patient",
    "spacy_ore", "openie_ore",
    "tagore_positive", "tagore_mutation", "tagore_mutations", "tagore_glof",
    "gpt_paper", "gpt_triplet",
]


"""
dataset
"""


def read_co_paper_mesh_to_gold_gene_from_db_pubmedkb_dataset(db_pubmedkb_file, write_log=False):
    mesh_to_gene = {}
    gene_set = set()

    with open(db_pubmedkb_file, "r", encoding="utf8") as f:
        for line in f:
            mesh, gold_gene_list, gene_ann_score = json.loads(line)

            if not gold_gene_list:
                continue

            # skip diseases that do not co-occur in a paper with any gold genes
            for gene in gold_gene_list:
                papers = gene_ann_score.get(gene, {}).get("paper", 0)
                if papers > 0:
                    break
            else:
                continue

            mesh_to_gene[mesh] = set(gold_gene_list)

            for gene in gold_gene_list:
                gene_set.add(gene)

    if write_log:
        meshs = len(mesh_to_gene)
        genes = len(gene_set)
        pairs = sum(len(g_set) for g_set in mesh_to_gene.values())
        logger.info(f"[clinvar2023 co-paper-disease] {pairs:,} pairs: {meshs:,} MeSHs, {genes:,} gold genes")

    return mesh_to_gene


def read_has_gold_gene_mesh_gene_annotation_from_db_pubmedkb_dataset(db_pubmedkb_file, write_log=False):
    mesh_gene_ann_score = {}
    gene_set = set()

    with open(db_pubmedkb_file, "r", encoding="utf8") as f:
        for line in f:
            mesh, gold_gene_list, gene_ann_score = json.loads(line)

            if not gold_gene_list:
                continue
            if not gene_ann_score:
                continue

            labeled_gene_ann_score = {}
            gold_gene_set = set(gold_gene_list)
            has_gold_gene = False

            for gene, ann_to_score in gene_ann_score.items():
                papers = ann_to_score.get("paper", 0)
                if papers <= 0:
                    continue

                assert "gold" not in ann_to_score
                if gene in gold_gene_set:
                    ann_to_score["gold"] = 1
                    has_gold_gene = True
                else:
                    ann_to_score["gold"] = 0

                labeled_gene_ann_score[gene] = ann_to_score
                gene_set.add(gene)

            if not has_gold_gene:
                continue

            mesh_gene_ann_score[mesh] = labeled_gene_ann_score

    if write_log:
        meshs = len(mesh_gene_ann_score)
        genes = len(gene_set)
        pairs = sum(len(gene_ann_score) for gene_ann_score in mesh_gene_ann_score.values())
        logger.info(f"[pubmedkb has-gold-gene-disease] {pairs:,} pairs: {meshs:,} MeSHs, {genes:,} co-paper genes")

    return mesh_gene_ann_score


def read_mesh_gene_embedding(embedding_file, embedding_type, write_log=False):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)

    mesh_gene_vector = defaultdict(lambda: {})
    gene_set = set()

    for mesh, gene, embedding_type_to_vector in data:
        mesh_gene_vector[mesh][gene] = embedding_type_to_vector[embedding_type]
        gene_set.add(gene)

    if write_log:
        meshs = len(mesh_gene_vector)
        genes = len(gene_set)
        pairs = sum(len(gene_to_vector) for gene_to_vector in mesh_gene_vector.values())
        logger.info(f"[mesh-gene embedding] {pairs:,} MeSH-gene pairs: {meshs:,} unique MeSHs, {genes:,} unique genes")

    return mesh_gene_vector


def get_map_recall(db_pubmedkb_file, prediction_file, mesh_gene_prediction=None):
    mesh_to_gold_gene_set = read_co_paper_mesh_to_gold_gene_from_db_pubmedkb_dataset(db_pubmedkb_file)

    if mesh_gene_prediction is None:
        with open(prediction_file, "rb") as f:
            mesh_gene_prediction = pickle.load(f)

    # recall
    golds = 0
    predicted_golds = 0

    # mean average precision (MAP)
    mean_ap = 0  # the MAP of the diseases that have predicted gold DG pairs
    total_ranked_lists = 0  # number of the diseases that have predicted gold DG pairs

    for mesh, gold_gene_set in mesh_to_gold_gene_set.items():
        golds += len(gold_gene_set)
        has_predicted_golds = False
        gold_list = []
        prediction_list = []

        for gene, prediction in mesh_gene_prediction.get(mesh, {}).items():
            if gene in gold_gene_set:
                has_predicted_golds = True
                gold_list.append(1)
            else:
                gold_list.append(0)
            prediction_list.append(prediction)

        if not has_predicted_golds:
            continue

        this_disease_predicted_golds = sum(gold_list)
        predicted_golds += this_disease_predicted_golds

        ap = average_precision_score(gold_list, prediction_list, pos_label=1)
        mean_ap += ap
        total_ranked_lists += 1

    mean_ap /= total_ranked_lists
    recall = predicted_golds / golds

    return mean_ap, recall


def max_min_normalization(a):
    m = a.min(axis=0)
    s = a.max(axis=0) - m
    for i in range(s.shape[0]):
        if s[i] == 0:
            s[i] = 1
    b = (a - m) / s
    return b


"""
method: use feature score as prediction score
"""


def extract_method_feature_prediction(db_pubmedkb_file, feature, prediction_file):
    logger.info(f"[method-feature-{feature}] getting prediction...")

    mesh_gene_ann_score = read_has_gold_gene_mesh_gene_annotation_from_db_pubmedkb_dataset(db_pubmedkb_file)
    mesh_gene_prediction = {}
    f_list = feature.split("+")

    for mesh, gene_ann_score in mesh_gene_ann_score.items():
        gene_to_prediction = {}

        for gene, ann_to_score in gene_ann_score.items():
            prediction = sum(ann_to_score.get(f, 0) for f in f_list)
            if prediction:
                gene_to_prediction[gene] = prediction

        if gene_to_prediction:
            mesh_gene_prediction[mesh] = gene_to_prediction

    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def extract_odds_ratio_feature(or_list):
    threshold_list = [0, 1, 5, 10]
    threshold_to_count = {threshold: 0 for threshold in threshold_list}

    log_odds_ratio_list = []
    log_mean = 0
    percentile_list = [25, 50, 75]
    percentile_to_log_or = {percentile: 0 for percentile in percentile_list}

    for odds_ratio in or_list:
        odds_ratio = float(odds_ratio)
        if odds_ratio <= 0:
            continue

        for threshold in threshold_list:
            if odds_ratio >= threshold:
                threshold_to_count[threshold] += 1
            else:
                break

        log_odds_ratio = np.log(odds_ratio)
        log_odds_ratio_list.append(log_odds_ratio)

    if log_odds_ratio_list:
        log_odds_ratio_array = np.array(log_odds_ratio_list, dtype=np.float64)
        log_mean = log_odds_ratio_array.mean()
        percentile_log_or_array = np.percentile(log_odds_ratio_array, percentile_list)
        for percentile, log_odds_ratio in zip(percentile_list, percentile_log_or_array):
            percentile_to_log_or[percentile] = log_odds_ratio

    return threshold_to_count, log_mean, percentile_to_log_or


def extract_method_odds_ratio_feature_prediction(db_pubmedkb_file, feature, prediction_file):
    logger.info(f"[method-feature-{feature}] getting prediction...")

    mesh_gene_ann_score = read_has_gold_gene_mesh_gene_annotation_from_db_pubmedkb_dataset(db_pubmedkb_file)
    mesh_gene_prediction = {}

    for mesh, gene_ann_score in mesh_gene_ann_score.items():
        gene_to_prediction = {}

        for gene, ann_to_score in gene_ann_score.items():
            or_list = ann_to_score.get("or_list", [])
            if not or_list:
                continue

            threshold_to_count, log_mean, percentile_to_log_or = extract_odds_ratio_feature(or_list)
            pos_count = threshold_to_count[0]

            if feature.startswith("or_count_"):
                threshold = feature[len("or_count_"):]
                threshold = int(threshold)
                count = threshold_to_count[threshold]
                if count > 0:
                    gene_to_prediction[gene] = count

            elif feature == "or_log_mean":
                if pos_count > 0:
                    gene_to_prediction[gene] = log_mean

            elif feature.startswith("or_percentile_"):
                if pos_count > 0:
                    percentile = feature[len("or_percentile_"):]
                    percentile = int(percentile)
                    log_or = percentile_to_log_or[percentile]
                    gene_to_prediction[gene] = log_or

        if gene_to_prediction:
            mesh_gene_prediction[mesh] = gene_to_prediction

    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def run_feature_method(db_pubmedkb_file, prediction_dir):
    assert "zero_equals_none" not in db_pubmedkb_file

    prediction_dir = os.path.join(prediction_dir, "feature")
    os.makedirs(prediction_dir, exist_ok=True)

    method_file_list = []

    # feature prediction
    for feature in ["gold"] + feature_list + ["sort_score"]:
        prediction_file = os.path.join(prediction_dir, f"{feature}.pkl")
        extract_method_feature_prediction(db_pubmedkb_file, feature, prediction_file)
        method_file_list.append((f"method-feature-{feature}", prediction_file))

    # odds ratio feature prediction
    or_count_list = [f"or_count_{i}" for i in [0]]
    or_percentile_list = [f"or_percentile_{i}" for i in []]
    or_feature_list = or_count_list + ["or_log_mean"] + or_percentile_list

    for feature in or_feature_list:
        prediction_file = os.path.join(prediction_dir, f"{feature}.pkl")
        extract_method_odds_ratio_feature_prediction(db_pubmedkb_file, feature, prediction_file)
        method_file_list.append((f"method-feature-{feature}", prediction_file))

    # evaluation
    for method, prediction_file in method_file_list:
        mean_ap, recall = get_map_recall(db_pubmedkb_file, prediction_file)
        logger.info(f"[{method: <55}] map={mean_ap: >5.1%} recall={recall: >5.1%}")
    return


"""
method: embedding ridge
"""


def get_pure_emb_ridge_feature(db_pubmedkb_file, embedding_file, embedding_type):
    mesh_gene_ann_score = read_has_gold_gene_mesh_gene_annotation_from_db_pubmedkb_dataset(db_pubmedkb_file)
    mesh_gene_vector = read_mesh_gene_embedding(embedding_file, embedding_type)
    data = []

    for mesh, gene_ann_score in mesh_gene_ann_score.items():
        gene_to_vector = mesh_gene_vector.get(mesh, {})
        if not gene_to_vector:
            continue

        gene_list = []
        y_array = []
        x_array = []

        for gene, ann_to_score in gene_ann_score.items():
            if gene not in gene_to_vector:
                continue
            gene_list.append(gene)

            y = ann_to_score["gold"]
            y_array.append(y)

            x = gene_to_vector[gene]
            x_array.append(x)

        y_array = np.array(y_array, dtype=np.float32)
        x_array = np.array(x_array, dtype=np.float32)

        data.append((mesh, gene_list, y_array, x_array))

    return data


def extract_method_pure_emb_ridge_prediction(db_pubmedkb_file, prediction_file, arg):
    from sklearn.linear_model import Ridge
    method = f"method-pure-emb-{arg.embedding_type}-ridge"

    logger.info(f"[{method}] reading feature...")
    data = get_pure_emb_ridge_feature(db_pubmedkb_file, arg.embedding_file, arg.embedding_type)

    logger.info(f"[{method}] getting prediction...")
    mesh_gene_prediction = {}
    splits = len(data)

    for test_split_index in range(splits):
        # data
        mesh, gene_list, _y_test, x_test = data[test_split_index]

        y_train = []
        x_train = []
        for split_index, (_split_mesh, _split_gene_list, split_y, split_x) in enumerate(data):
            if split_index == test_split_index:
                continue
            y_train.append(split_y)
            x_train.append(split_x)
        y_train = np.concatenate(y_train, axis=0)
        x_train = np.concatenate(x_train, axis=0)

        # model
        model = Ridge(alpha=1, solver="svd")
        model.fit(x_train, y_train)
        prediction_array = model.predict(x_test)

        mesh_gene_prediction[mesh] = dict(zip(gene_list, prediction_array))
        predicted_splits = test_split_index + 1
        logger.info(f"[{method}] predicted {predicted_splits:,}/{splits:,}")

    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def run_pure_emb_ridge_method(db_pubmedkb_file, prediction_dir, arg):
    method = f"method-pure-emb-{arg.embedding_type}-ridge"
    logger.info(f"{method}: start")
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_file = os.path.join(prediction_dir, f"pure_ridge_{arg.embedding_type}_{arg.embedding_dimension}.pkl")

    extract_method_pure_emb_ridge_prediction(db_pubmedkb_file, prediction_file, arg)
    mean_ap, recall = get_map_recall(db_pubmedkb_file, prediction_file)

    logger.info(f"[{method: <55}] map={mean_ap: >5.1%} recall={recall: >5.1%}")
    logger.info(f"{method}: done")
    return


"""
method: lambda + gradient-boosted decision trees
"""


def get_stats_emb_lightgbm_feature(db_pubmedkb_file, embedding_file, embedding_type, embedding_dimension):
    mesh_gene_ann_score = read_has_gold_gene_mesh_gene_annotation_from_db_pubmedkb_dataset(db_pubmedkb_file)
    mesh_gene_vector = read_mesh_gene_embedding(embedding_file, embedding_type)
    data = []

    def add_gene():
        gene_list.append(gene)
        y_array.append(y)

        x = []
        for feature in feature_list:
            x.append(ann_to_score.get(feature, 0))

        or_list = ann_to_score.get("or_list", [])
        threshold_to_count, _log_mean, _percentile_to_log_or = extract_odds_ratio_feature(or_list)
        x.append(threshold_to_count[0])

        x_array.append(x)

        e = gene_to_vector.get(gene, np.zeros(embedding_dimension, dtype=np.float64))
        e_array.append(e)
        return

    for mesh, gene_ann_score in mesh_gene_ann_score.items():
        gene_to_vector = mesh_gene_vector.get(mesh, {})

        gene_list = []
        y_array = []
        x_array = []
        e_array = []

        genes = len(gene_ann_score)

        for gene, ann_to_score in gene_ann_score.items():
            y = ann_to_score["gold"]
            if y == 0:
                continue
            add_gene()

        pos_genes = len(gene_list)

        for gene, ann_to_score in gene_ann_score.items():
            y = ann_to_score["gold"]
            if y == 1:
                continue
            add_gene()

        if genes > 10000:
            di = len(data) + 1
            logger.info(f"#{di:,} mesh={mesh} genes={genes:,} pos_genes={pos_genes:,}")

        y_array = np.array(y_array, dtype=np.float32)

        x_array = np.array(x_array, dtype=np.float32)
        x2_array = max_min_normalization(x_array)
        x3_array = np.array(e_array, dtype=np.float32)
        x_array = np.concatenate((x_array, x2_array, x3_array), axis=1)

        data.append((mesh, gene_list, y_array, x_array))

    return data


def extract_method_stats_emb_lightgbm_prediction(db_pubmedkb_file, prediction_file, arg):
    import lightgbm as lgb
    method = f"method-stats-emb-lgb-{arg.name}"

    logger.info(f"[{method}] reading feature...")
    data = get_stats_emb_lightgbm_feature(
        db_pubmedkb_file, arg.embedding_file, arg.embedding_type, arg.embedding_dimension,
    )

    logger.info(f"[{method}] getting prediction...")
    mesh_gene_prediction = {}
    splits = len(data)

    max_genes = 10000  # lightgbm limitation

    for test_split_index in range(splits):
        # data
        mesh, gene_list, _y_test, x_test = data[test_split_index]

        y_train = []
        x_train = []
        q_train = []
        for split_index, (_split_mesh, _split_gene_list, split_y, split_x) in enumerate(data):
            if split_index == test_split_index:
                continue
            if split_y.shape[0] > max_genes:
                split_y = split_y[:max_genes]
                split_x = split_x[:max_genes]
            y_train.append(split_y)
            x_train.append(split_x)
            q_train.append(split_y.shape[0])
        y_train = np.concatenate(y_train, axis=0)
        x_train = np.concatenate(x_train, axis=0)

        # model
        model = lgb.LGBMRanker(
            objective="lambdarank",
            device="cpu",
            verbosity=-1,  # log level: fatal
            force_row_wise=True,  # build histogram data-point-wise instead of feature-wise
            data_sample_strategy=arg.lgb_data_sample_strategy,
            num_leaves=arg.lgb_num_leaves,
            max_depth=arg.lgb_max_depth,
        )
        model.fit(x_train, y_train, group=q_train)

        prediction_array = model.predict(x_test)
        mesh_gene_prediction[mesh] = dict(zip(gene_list, prediction_array))

        predicted_splits = test_split_index + 1
        logger.info(f"[{method}] predicted {predicted_splits:,}/{splits:,}")

    logger.info(f"[{method}] saving prediction to file...")
    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def run_stats_emb_lightgbm_method(db_pubmedkb_file, prediction_dir, arg):
    prediction_dir = os.path.join(prediction_dir, "stats_emb_lgb")
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_file = os.path.join(prediction_dir, f"{arg.name}.pkl")

    extract_method_stats_emb_lightgbm_prediction(db_pubmedkb_file, prediction_file, arg)
    mean_ap, recall = get_map_recall(db_pubmedkb_file, prediction_file)

    method = f"method-stats-emb-lgb-{arg.name}"
    logger.info(f"[{method: <55}] map={mean_ap: >5.1%} recall={recall: >5.1%}")
    return

