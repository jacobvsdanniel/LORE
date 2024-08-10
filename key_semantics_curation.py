import os
import csv
import json
import pickle
import logging
from collections import defaultdict

from ml_ranker import get_map_recall, read_co_paper_mesh_to_gold_gene_from_db_pubmedkb_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
csv.register_dialect(
    "csv", delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True,
    escapechar=None, lineterminator="\n", skipinitialspace=False,
)


def read_csv(file, dialect, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        row_list = [row for row in reader]

    if write_log:
        rows = len(row_list)
        logger.info(f"Read {rows:,} rows")
    return row_list


def extract_gpt_triplet_tokens(triplet_file, triplet_token_file, dataset_file):
    from nltk.tokenize.destructive import NLTKWordTokenizer
    from nltk.stem.snowball import SnowballStemmer

    tokenizer = NLTKWordTokenizer()
    stemmer = SnowballStemmer("english")

    lines = 0
    mesh_gene_ann_score = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    token_set = set()

    with open(triplet_file, "r", encoding="utf8") as fr, \
         open(triplet_token_file, "w", encoding="utf8") as fw:
        for line in fr:
            lines += 1
            _p, hrt, d_name_count, g_name_count, _v_name_count = json.loads(line)

            mesh = next(iter(d_name_count))
            gene = next(iter(g_name_count))
            name_set = set()
            ann_to_score = mesh_gene_ann_score[mesh][gene]

            for id_name_count in [d_name_count, g_name_count]:
                for _id, name_to_count in id_name_count.items():
                    for name in name_to_count:
                        name_set.add(name)

            token_list = tokenizer.tokenize(" ".join(hrt))
            lemma_list = []
            for token in token_list:
                if len(token) <= 2:
                    continue
                if token in name_set:
                    continue
                token = stemmer.stem(token)
                lemma_list.append(token)
                ann_to_score[token] += 1
                token_set.add(token)

            datum = [_p, hrt, d_name_count, g_name_count, _v_name_count, lemma_list]
            json.dump(datum, fw)
            fw.write("\n")

            if lines % 100000 == 0:
                meshs = len(mesh_gene_ann_score)
                pairs = sum(len(gene_ann_score) for gene_ann_score in mesh_gene_ann_score.values())
                tokens = len(token_set)
                logger.info(f"{lines:,} lines: {meshs:,} MeSHs; {pairs:,} pairs; {tokens:,} tokens")
    if lines % 100000 != 0:
        meshs = len(mesh_gene_ann_score)
        pairs = sum(len(gene_ann_score) for gene_ann_score in mesh_gene_ann_score.values())
        tokens = len(token_set)
        logger.info(f"{lines:,} lines: {meshs:,} MeSHs; {pairs:,} pairs; {tokens:,} tokens")

    with open(dataset_file, "w", encoding="utf8") as f:
        for mesh, gene_ann_score in mesh_gene_ann_score.items():
            datum = [mesh, [], gene_ann_score]
            json.dump(datum, f)
            f.write("\n")

    logger.info("done")
    return


class TokenStats:
    def __init__(self, token):
        self.token = token

        self.gold_dgs = 0
        self.dgs = 0

        self.gold_score = 0
        self.score = 0
        return


def extract_token_stats(db_pubmedkb_file, token_dataset_file, token_stats_file):
    mesh_to_gold_gene_set = read_co_paper_mesh_to_gold_gene_from_db_pubmedkb_dataset(
        db_pubmedkb_file, write_log=True,
    )

    lines = 0
    meshs = 0
    token_to_stats = {}

    logger.info(f"reading {token_dataset_file}")
    with open(token_dataset_file, "r", encoding="utf8") as f:
        for line in f:
            lines += 1
            mesh, _, gene_token_score = json.loads(line)
            gold_gene_set = mesh_to_gold_gene_set.get(mesh, None)
            if gold_gene_set is None:
                continue
            meshs += 1

            for gene, token_to_score in gene_token_score.items():
                for token, score in token_to_score.items():
                    token_stats = token_to_stats.get(token)
                    if token_stats is None:
                        token_stats = TokenStats(token)
                        token_to_stats[token] = token_stats
                    token_stats.dgs += 1
                    token_stats.score += score
                    if gene in gold_gene_set:
                        token_stats.gold_dgs += 1
                        token_stats.gold_score += score
    tokens = len(token_to_stats)
    logger.info(f"{lines:,} all MeSHs; has-gold-gene MeSHs: {meshs:,} MeSHs, {tokens:,} tokens")

    logger.info("ranking tokens...")
    token_stats_list = []
    score_list = []
    dgs_threshold = 50
    for token, token_stats in token_to_stats.items():
        if token_stats.dgs < dgs_threshold:
            continue
        token_stats_list.append(token_stats)
        gold_score_ratio = token_stats.gold_score / token_stats.score
        gold_dgs_ratio = token_stats.gold_dgs / token_stats.dgs
        score_list.append(
            (gold_score_ratio, token_stats.gold_dgs, gold_dgs_ratio)
        )
    index_list = sorted(range(len(token_stats_list)), key=lambda i: score_list[i], reverse=True)
    tokens = len(token_stats_list)
    logger.info(f"{tokens:,} tokens appears in at least {dgs_threshold:,} DGs")

    logger.info(f"writing {token_stats_file}")
    header = ["token", "gold_score_ratio", "gold_score", "score", "gold_dgs", "dgs"]
    with open(token_stats_file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, dialect="csv")
        writer.writerow(header)

        for index in index_list:
            token_stats = token_stats_list[index]
            gold_score_ratio = score_list[index][0]
            gold_score_ratio = f"{gold_score_ratio:.0%}"
            row = [
                token_stats.token, gold_score_ratio,
                token_stats.gold_score, token_stats.score, token_stats.gold_dgs, token_stats.dgs,
            ]
            writer.writerow(row)

    logger.info("done")
    return


def extract_token_to_triplet_data(db_pubmedkb_file, token_stats_file, triplet_token_file, token_to_triplet_file):
    # gold label
    mesh_to_gold_gene_set = read_co_paper_mesh_to_gold_gene_from_db_pubmedkb_dataset(
        db_pubmedkb_file, write_log=True,
    )

    # token stats
    token_stats_data = read_csv(token_stats_file, "csv")
    header, token_stats_data = token_stats_data[0], token_stats_data[1:]
    logger.info(f"header: {header}")
    token_to_stats = {
        token: (gold_score_ratio, gold_score, score, gold_dgs, dgs)
        for token, gold_score_ratio, gold_score, score, gold_dgs, dgs in token_stats_data
    }
    del token_stats_data

    token_label_mesh_gene_triplet = {
        token: {
            label: defaultdict(lambda: defaultdict(lambda: []))
            for label in [0, 1]
        }
        for token in token_to_stats
    }
    token_to_triplets = defaultdict(lambda: 0)

    # triplet data
    logger.info(f"reading {triplet_token_file}")
    lines = 0
    triplets = 0

    with open(triplet_token_file, "r", encoding="utf8") as f:
        for line in f:
            lines += 1
            p, hrt, d_name_count, g_name_count, _v_name_count, token_list = json.loads(line)
            mesh = next(iter(d_name_count))
            gold_gene_set = mesh_to_gold_gene_set.get(mesh)

            if gold_gene_set is not None:
                gene = next(iter(g_name_count))
                label = 1 if gene in gold_gene_set else 0

                for token in token_list:
                    if token not in token_to_stats:
                        continue
                    token_to_triplets[token] += 1
                    token_label_mesh_gene_triplet[token][label][mesh][gene].append(
                        (p, hrt)
                    )
                    triplets += 1

            if lines % 1000000 == 0:
                tokens = len(token_to_triplets)
                logger.info(f"{lines:,} lines: {triplets:,} triplets; {tokens:,} tokens")
        if lines % 1000000 != 0:
            tokens = len(token_to_triplets)
            logger.info(f"{lines:,} lines: {triplets:,} triplets; {tokens:,} tokens")

    logger.info(f"writing to {token_to_triplet_file}")
    lines = 0
    with open(token_to_triplet_file, "w", encoding="utf8") as f:
        for token, stats in token_to_stats.items():
            label_mesh_gene_triplet = token_label_mesh_gene_triplet[token]
            datum = {
                "token": token,
                "gold_score_ratio": stats[0],
                "gold_score": stats[1],
                "score": stats[2],
                "gold_dgs": stats[3],
                "dgs": stats[4],
                "triplets": token_to_triplets[token],
                "gold_mesh_gene_triplet": label_mesh_gene_triplet[1],
                "other_mesh_gene_triplet": label_mesh_gene_triplet[0],
            }
            json.dump(datum, f)
            f.write("\n")
            lines += 1
            if lines % 100 == 0:
                logger.info(f"{lines:,} lines")
        if lines % 100 != 0:
            logger.info(f"{lines:,} lines")

    logger.info("done")
    return


def curate_token_to_triplet_data(token_to_triplet_file, token_curation_file):
    import random

    dg_samples = 5
    lines = 0

    with open(token_to_triplet_file, "r", encoding="utf8") as fr, \
         open(token_curation_file, "a", encoding="utf8") as fw:
        writer = csv.writer(fw, dialect="csv")

        for line in fr:
            lines += 1

            datum = json.loads(line)
            token = datum["token"]
            gold_score_ratio = datum["gold_score_ratio"]
            gold_score = datum["gold_score"]
            score = datum["score"]
            gold_dgs = datum["gold_dgs"]
            dgs = datum["dgs"]
            triplets = datum["triplets"]

            prefix = f"#{lines:,} [{token}]"
            logger.info(f"{prefix} gold_score_ratio = {gold_score_ratio} = {gold_score}/{score}")
            logger.info(f"{prefix} gold_dg_ratio = {gold_dgs}/{dgs}")
            logger.info(f"{prefix} {triplets:,} triplets")

            for label in ["gold", "other"]:
                mesh_gene_triplet = datum[f"{label}_mesh_gene_triplet"]
                dg_list = [
                    (mesh, gene)
                    for mesh, gene_to_triplet in mesh_gene_triplet.items()
                    for gene in gene_to_triplet
                ]
                dg_list = random.sample(dg_list, dg_samples)
                for mesh, gene in dg_list:
                    p, (h, r, t) = random.choice(mesh_gene_triplet[mesh][gene])
                    logger.info(f"{prefix} [{label}] [PMID-{p}] [{h}] [{r}] [{t}]")

            annotation = input("Annotate: ")
            if not annotation:
                continue

            logger.info(f"{prefix} annotation={annotation}")
            writer.writerow([token, annotation])
            fw.flush()
            logger.info("")
            logger.info("")
            logger.info("")

    logger.info("done")
    return


def extract_top_token_dataset(token_dataset_file, token_stats_file, top_token_dataset_file):
    # read top tokens
    token_stats_data = read_csv(token_stats_file, "csv")
    token_stats_header, token_stats_data = token_stats_data[0], token_stats_data[1:]
    logger.info(f"token_stats_header: {token_stats_header}")
    del token_stats_header
    top_token_set = set(row[0] for row in token_stats_data)
    del token_stats_data
    top_tokens = len(top_token_set)
    logger.info(f"{top_tokens:,} top_tokens (tokens that occur in at least X DGs)")

    # extract D->G->top_token->score dataset
    logger.info(f"reading {token_dataset_file}")
    lines = 0
    meshs = 0
    pairs = 0

    with open(token_dataset_file, "r", encoding="utf8") as fr, \
         open(top_token_dataset_file, "w", encoding="utf8") as fw:
        for line in fr:
            lines += 1
            mesh, _, gene_token_score = json.loads(line)
            gene_token_score = {
                gene: {
                    token: score
                    for token, score in token_to_score.items()
                    if token in top_token_set
                }
                for gene, token_to_score in gene_token_score.items()
            }
            if gene_token_score:
                json.dump([mesh, _, gene_token_score], fw)
                fw.write("\n")
                meshs += 1
                pairs += len(gene_token_score)

    logger.info(f"{lines:,} all MeSHs; {meshs:,} has-top-token MeSHs; {pairs:,} has-top-token DG pairs")
    return


def extract_method_token_prediction(top_token_dataset_file, token_set, prediction_file):
    tokens = len(token_set)
    logger.info(f"[method-token-{tokens}] getting prediction...")

    mesh_gene_prediction = {}

    with open(top_token_dataset_file, "r", encoding="utf8") as f:
        for line in f:
            mesh, _, gene_token_score = json.loads(line)
            gene_to_prediction = {}
            for gene, token_to_score in gene_token_score.items():
                prediction = sum(
                    score
                    for token, score in token_to_score.items()
                    if token in token_set
                )
                if prediction:
                    gene_to_prediction[gene] = prediction
            if gene_to_prediction:
                mesh_gene_prediction[mesh] = gene_to_prediction

    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def run_token_method(db_pubmedkb_file, top_token_dataset_file, token_stats_file, token_curation_file, prediction_dir):
    prediction_dir = os.path.join(prediction_dir, "token")
    os.makedirs(prediction_dir, exist_ok=True)

    # token to gold_score_ratio
    token_stats_data = read_csv(token_stats_file, "csv")
    token_stats_header, token_stats_data = token_stats_data[0], token_stats_data[1:]
    logger.info(f"token_stats_header: {token_stats_header}")
    token_to_gold_score_ratio = {
        token: int(gold_score) / int(score)
        for token, gold_score_ratio, gold_score, score, gold_dgs, dgs in token_stats_data
    }
    top_tokens = len(token_to_gold_score_ratio)
    logger.info(f"{top_tokens:,} top_tokens")
    del token_stats_header
    del token_stats_data

    # token curation
    token_curation_data = read_csv(token_curation_file, "csv")
    curated_token_set = set(
        token
        for token, annotation in token_curation_data
        if annotation == "1"
    )
    curated_tokens = len(curated_token_set)
    logger.info(f"{curated_tokens:,} curated_tokens")
    del token_curation_data

    method_file_list = []

    # method: top token filtered by different gold_score_ratio
    for gold_score_ratio_threshold in range(1, 10):
        method = f"gold_score_ratio_threshold_0.{gold_score_ratio_threshold}"
        prediction_file = os.path.join(prediction_dir, f"{method}.pkl")
        method_file_list.append((method, prediction_file))

        gold_score_ratio_threshold /= 10
        token_set = set(
            token
            for token, gold_score_ratio in token_to_gold_score_ratio.items()
            if gold_score_ratio >= gold_score_ratio_threshold
        )
        extract_method_token_prediction(top_token_dataset_file, token_set, prediction_file)

    # method: top token filtered by gold_score_ratio at 0.5 and curation
    method = f"curated_token"
    prediction_file = os.path.join(prediction_dir, f"{method}.pkl")
    method_file_list.append((method, prediction_file))
    extract_method_token_prediction(top_token_dataset_file, curated_token_set, prediction_file)

    # evaluation
    for method, prediction_file in method_file_list:
        mean_ap, recall = get_map_recall(db_pubmedkb_file, prediction_file)
        logger.info(f"[{method: <55}] map={mean_ap: >5.1%} recall={recall: >5.1%}")
    return


def extract_curated_token_sample_triplet_data(token_to_triplet_file, token_curation_file, token_sample_triplet_file):
    import random

    # read curated_token_list
    token_curation_data = read_csv(token_curation_file, "csv")
    curated_token_list = [
        token
        for token, annotation in token_curation_data
        if annotation == "1"
    ]
    curated_tokens = len(curated_token_list)
    logger.info(f"{curated_tokens:,} curated_tokens")
    del token_curation_data

    # extract triplet samples
    dg_samples = 10

    with open(token_to_triplet_file, "r", encoding="utf8") as fr, \
         open(token_sample_triplet_file, "w", encoding="utf8") as fw:

        for line in fr:
            datum = json.loads(line)
            token = datum["token"]
            if token not in curated_token_list:
                continue

            sample_list = []

            for label in ["gold", "other"]:
                mesh_gene_triplet = datum[f"{label}_mesh_gene_triplet"]
                dg_list = [
                    (mesh, gene)
                    for mesh, gene_to_triplet in mesh_gene_triplet.items()
                    for gene in gene_to_triplet
                ]
                dg_list = random.sample(dg_list, dg_samples)
                for mesh, gene in dg_list:
                    p, (h, r, t) = random.choice(mesh_gene_triplet[mesh][gene])
                    sample_list.append([label, mesh, gene, p, f"{h} {r} {t}"])

            out_datum = {
                "token": token,
                "gold_score": int(datum["gold_score"]),
                "score": int(datum["score"]),
                "gold_dgs": int(datum["gold_dgs"]),
                "dgs": int(datum["dgs"]),
                "triplets": datum["triplets"],
                "sample_list": sample_list,
            }

            json.dump(out_datum, fw)
            fw.write("\n")

    logger.info("done")
    return


def curate_token_explanation(token_sample_triplet_file, token_explanation_file):
    lines = 0

    with open(token_sample_triplet_file, "r", encoding="utf8") as fr, \
         open(token_explanation_file, "a", encoding="utf8") as fw:
        writer = csv.writer(fw, dialect="csv")

        for line in fr:
            lines += 1
            datum = json.loads(line)
            token = datum["token"]
            sample_list = datum["sample_list"]
            for label, _mesh, _gene, p, triplet in sample_list:
                logger.info(f"#{lines} [{token}] [{label}] [PMID-{p}] {triplet}")
            explanation = input("explanation: ")
            if explanation:
                writer.writerow([token, explanation])

    return


class TokenSemanticSubClass:
    def __init__(self):
        self.id = None
        self.name = None
        self.token_to_explanation = {}
        return

    def extract_sample_data(self, token_to_sample_list, curation_sample_file):
        full_file = curation_sample_file + " full.csv"
        concise_file = curation_sample_file + " concise.csv"

        full_header = ["key token", "key semantics", "pair-in-clinvar", "disease", "gene", "PMID", "triplet"]
        concise_header = ["key semantics", "PMID", "triplet"]

        with open(full_file, "w", encoding="utf8") as f_full, \
             open(concise_file, "w", encoding="utf8") as f_concise:

            full_writer = csv.writer(f_full, dialect="csv")
            concise_writer = csv.writer(f_concise, dialect="csv")

            full_writer.writerow(full_header)
            concise_writer.writerow(concise_header)

            for token, explanation in self.token_to_explanation.items():
                for label, mesh, gene, pmid, triplet in token_to_sample_list[token]:
                    gene = f"GENE:{gene}"
                    pmid = f"PMID:{pmid}"
                    full_writer.writerow([token, explanation, label, mesh, gene, pmid, triplet])
                    concise_writer.writerow([explanation, pmid, triplet])
        return

    def extract_token_to_id(self, token_to_id, main_class_id):
        for token in self.token_to_explanation:
            token_to_id[token] = (main_class_id, self.id)
        return


class TokenSemanticMainClass:
    def __init__(self):
        self.id = None
        self.name = None
        self.sub_class_id_to_data = {}
        return

    def read_data(self, curation_file):
        sub_class = None
        sub_class_id = 0

        with open(curation_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("["):
                    assert sub_class is None
                    sub_class_id += 1
                    sub_class = TokenSemanticSubClass()
                    sub_class.id = sub_class_id
                    sub_class.name = line[1:-1]

                elif line:
                    i = line.find(",")
                    assert i != -1
                    token, explanation = line[:i], line[i + 1:]
                    sub_class.token_to_explanation[token] = explanation

                else:
                    assert sub_class is not None
                    self.sub_class_id_to_data[sub_class.id] = sub_class
                    sub_class = None

            assert sub_class is not None
            self.sub_class_id_to_data[sub_class.id] = sub_class
        return

    def extract_sample_data(self, token_to_sample_list, curation_sample_dir):
        os.makedirs(curation_sample_dir, exist_ok=True)

        for sub_class_id, sub_class in self.sub_class_id_to_data.items():
            sub_class_file = os.path.join(curation_sample_dir, f"{self.id}.{sub_class_id} {sub_class.name}")
            sub_class.extract_sample_data(token_to_sample_list, sub_class_file)
        return

    def extract_token_to_id(self, token_to_id):
        for sub_class_id, sub_class in self.sub_class_id_to_data.items():
            sub_class.extract_token_to_id(token_to_id, self.id)
        return


class TokenSemanticData:
    def __init__(self):
        self.main_class_id_to_data = {}
        return

    def read_data(self, curation_dir):
        curation_file_list = os.listdir(curation_dir)
        curation_file_list = sorted(curation_file_list)

        for file_name in curation_file_list:
            main_class = TokenSemanticMainClass()
            main_class.id = int(file_name[0])
            main_class.name = file_name[2:-4]
            curation_file = os.path.join(curation_dir, file_name)
            main_class.read_data(curation_file)
            self.main_class_id_to_data[main_class.id] = main_class
        return

    def extract_sample_data(self, token_to_sample_list, curation_sample_dir):
        os.makedirs(curation_sample_dir, exist_ok=True)

        for main_class_id, main_class in self.main_class_id_to_data.items():
            main_class_dir = os.path.join(curation_sample_dir, f"{main_class_id} {main_class.name}")
            main_class.extract_sample_data(token_to_sample_list, main_class_dir)
        return

    def extract_token_to_id(self, token_to_id):
        for main_class_id, main_class in self.main_class_id_to_data.items():
            main_class.extract_token_to_id(token_to_id)
        return


def check_semantic_curation(token_explanation_file, curation_dir):
    # all tokens and explanations
    token_explanation_list = read_csv(token_explanation_file, "csv")
    token_to_explanation = {
        token: explanation
        for token, explanation in token_explanation_list
    }
    assert len(token_to_explanation) == len(token_explanation_list)
    del token_explanation_list

    # all hierarchical curation
    token_semantic_data = TokenSemanticData()
    token_semantic_data.read_data(curation_dir)

    # check
    for main_class_id, main_class in token_semantic_data.main_class_id_to_data.items():
        logger.info(f"{main_class_id} {main_class.name}")

        for sub_class_id, sub_class in main_class.sub_class_id_to_data.items():
            logger.info(f"{main_class_id}.{sub_class_id} {sub_class.name}")

            for token, explanation in sub_class.token_to_explanation.items():
                logger.info(f"[{token}] {explanation}")
                assert token_to_explanation[token] == explanation
                del token_to_explanation[token]

    assert len(token_to_explanation) == 0
    return


def extract_curation_sample_data(curation_dir, token_sample_triplet_file, curation_sample_dir):
    token_semantic_data = TokenSemanticData()
    token_semantic_data.read_data(curation_dir)

    token_to_sample_list = {}
    with open(token_sample_triplet_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            token = datum["token"]
            sample_list = datum["sample_list"]
            token_to_sample_list[token] = sample_list

    token_semantic_data.extract_sample_data(token_to_sample_list, curation_sample_dir)
    return

