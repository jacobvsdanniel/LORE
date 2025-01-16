import os
import re
import csv
import sys
import json
import math
import time
import heapq
import pickle
import random
import struct
import asyncio
import difflib
import logging
import argparse
import datetime
import traceback
from collections import defaultdict

import numpy as np

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


"""
Config
"""


class Config:
    def __init__(self):
        self.task = ""

        # LLM-ORE
        self.literature_entity_file = ""
        self.literature_content_file = ""
        self.ore_extraction_file = ""
        self.ore_prompt_file = ""
        self.ore_knowledge_graph_file = ""
        self.ore_server = ""
        self.ore_model = ""
        self.ore_concurrent_requests = 0
        self.ore_requests_per_minute = 0
        self.ore_tokens_per_minute = 0
        self.ore_generations_per_prompt = 3
        self.ore_max_title_tokens = 200
        self.ore_max_article_tokens = 2500
        self.ore_max_completion_tokens = 200
        self.ore_max_prompt_runs = 10
        self.ore_sort_knowledge_graph = True
        self.ore_max_relation_similarity = 0.9

        # LLM-EMB
        self.emb_meta_file = ""
        self.emb_bytes_file = ""
        self.emb_file = ""
        self.emb_model = ""
        self.emb_dimension = 0
        self.emb_max_text_tokens = 8191
        self.emb_requests_per_minute = 0
        self.emb_tokens_per_minute = 0
        self.emb_max_prompt_runs = 10

        # ML-Ranker
        self.ranker_entity_feature_file = ""
        self.ranker_entity_label_file = ""
        self.ranker_D_split_file = ""
        self.ranker_entity_score_file = ""
        self.ranker_model_file = ""
        self.ranker_lgb_data_sample_strategy = "goss"
        self.ranker_lgb_num_leaves = 12
        self.ranker_lgb_max_depth = 4

        # Key-Semantics
        self.semantics_lemma_file = ""
        self.semantics_candidate_file = ""
        self.semantics_file = ""
        self.semantics_taxonomy_path = ""
        self.semantics_knowledge_graph_file = ""
        self.semantics_min_DGs = 100
        self.semantics_min_gold_DG_relations = 0.5
        self.semantics_samples_per_lemma = 10
        return

    def load(self, config_file):
        with open(config_file, "r", encoding="utf8") as f:
            parameter_to_value = json.load(f)

        for parameter, value in parameter_to_value.items():
            setattr(self, parameter, value)
            logger.info(f"[config.{parameter}] {value}")

        return


config = Config()


"""
LLM-ORE
"""


class DeepInfraTaskDatum:
    def __init__(
            self, prompt_id,
            d, g, p, title, article,
            d_name, g_name, d_alias_list, g_alias_list,
            template,
    ):
        self.runs = 0
        self.prompt_id = prompt_id

        self.d = d
        self.g = g
        self.p = p
        self.title = title
        self.article = article

        self.d_name = d_name
        self.g_name = g_name
        self.d_alias_list = d_alias_list
        self.g_alias_list = g_alias_list

        self.text_in = template.replace(
            "yoloTITLEyolo", self.title,
        ).replace(
            "yoloTEXTyolo", self.article,
        ).replace(
            "yoloGENEyolo", self.g_name,
        ).replace(
            "yoloDISEASEyolo", self.d_name,
        )
        self.text_out_list = []

        self.request_start_time = 0
        self.request_end_time = 0

        self.log_string = (
            f"[#{self.prompt_id}]"
            f" [D:{self.d}]"
            f" [G:{self.g}]"
            f" [P:{self.p}]"
            f" [HEAD:{self.g_name}]"
            f" [TAIL:{self.d_name}]"
        )
        return

    def get_json_obj(self):
        request_start_time = datetime.datetime.fromtimestamp(self.request_start_time).isoformat()
        request_end_time = datetime.datetime.fromtimestamp(self.request_end_time).isoformat()

        json_obj = {
            "prompt_id": self.prompt_id,
            "D_id": self.d, "G_id": self.g, "P_id": self.p,
            "D_name": self.d_name, "G_name": self.g_name,
        }

        if self.d_alias_list:
            json_obj["D_alias_list"] = self.d_alias_list
        if self.g_alias_list:
            json_obj["G_alias_list"] = self.g_alias_list

        json_obj["text_out_list"] = self.text_out_list
        json_obj["request_start_time"] = request_start_time
        json_obj["request_end_time"] = request_end_time
        return json_obj


class OpenAITaskDatum:
    def __init__(
            self, prompt_id,
            d, g, p, title, article,
            d_name, g_name, d_alias_list, g_alias_list,
            template, tokenizer,
    ):
        self.runs = 0
        self.prompt_id = prompt_id

        self.d = d
        self.g = g
        self.p = p
        self.title = title
        self.article = article

        self.d_name = d_name
        self.g_name = g_name
        self.d_alias_list = d_alias_list
        self.g_alias_list = g_alias_list

        self.text_in = template.replace(
            "yoloTITLEyolo", self.title,
        ).replace(
            "yoloTEXTyolo", self.article,
        ).replace(
            "yoloGENEyolo", self.g_name,
        ).replace(
            "yoloDISEASEyolo", self.d_name,
        )
        self.in_tokens = len(tokenizer.encode(self.text_in))
        self.text_out_list = []

        self.request_start_time = 0
        self.request_end_time = 0

        self.log_string = (
            f"[#{self.prompt_id}]"
            f" [D:{self.d}]"
            f" [G:{self.g}]"
            f" [P:{self.p}]"
            f" [HEAD:{self.g_name}]"
            f" [TAIL:{self.d_name}]"
        )
        return

    def get_json_obj(self):
        request_start_time = datetime.datetime.fromtimestamp(self.request_start_time).isoformat()
        request_end_time = datetime.datetime.fromtimestamp(self.request_end_time).isoformat()

        json_obj = {
            "prompt_id": self.prompt_id,
            "D_id": self.d, "G_id": self.g, "P_id": self.p,
            "D_name": self.d_name, "G_name": self.g_name,
        }

        if self.d_alias_list:
            json_obj["D_alias_list"] = self.d_alias_list
        if self.g_alias_list:
            json_obj["G_alias_list"] = self.g_alias_list

        json_obj["in_tokens"] = self.in_tokens
        json_obj["text_out_list"] = self.text_out_list
        json_obj["request_start_time"] = request_start_time
        json_obj["request_end_time"] = request_end_time
        return json_obj


async def ore_request(client, task_datum):
    task_datum.runs += 1
    task_datum.request_start_time = time.time()
    completion = await client.chat.completions.create(
        model=config.ore_model,
        n=config.ore_generations_per_prompt,
        messages=[
            {"role": "user", "content": task_datum.text_in},
        ],
        max_completion_tokens=config.ore_max_completion_tokens,
    )
    task_datum.request_end_time = time.time()

    task_datum.text_out_list = [
        choice.message.content
        for choice in completion.choices
    ]
    return task_datum


async def run_ore_extraction_deepinfra():
    from openai import AsyncOpenAI

    # set up client
    logger.info("setting up ORE client...")
    api_key = input("please input ORE server API key: ")
    logger.info("received server API key")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.ore_server,  # for Deep Infra: "https://api.deepinfra.com/v1/openai"
    )

    # set up task management
    requests_quota = config.ore_concurrent_requests
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # read prompt template
    with open(config.ore_prompt_file, "r", encoding="utf8") as f:
        template = f.read()

    # read completed data
    completed_prompt_id_set = set()
    if os.path.exists(config.ore_extraction_file):
        logger.info("reading completed data...")
        with open(config.ore_extraction_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_prompt_id_set.add(datum["prompt_id"])
        completed_prompts = len(completed_prompt_id_set)
        logger.info(f"read {completed_prompts:,} completed_prompts")

    # read article data
    logger.info("reading article data...")
    p_to_title_article = {}
    with open(config.literature_content_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            p = datum["P_id"]
            title = datum["title"]
            article = datum["article"]
            p_to_title_article[p] = (title, article)
    articles = len(p_to_title_article)
    logger.info(f"read {articles:,} articles")

    # run extraction by calling server
    logger.info("running extraction...")
    with open(config.literature_entity_file, "r", encoding="utf8") as fr, \
         open(config.ore_extraction_file, "a", encoding="utf8") as fw:

        for li, line in enumerate(fr):
            prompt_id = li + 1

            if prompt_id in completed_prompt_id_set:
                logger.info(f"skip: [#{prompt_id}]")
                continue

            # create task datum
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            p = datum["P_id"]
            d_name = datum["D_name"]
            g_name = datum["G_name"]
            d_alias_list = datum.get("D_alias_list", [])
            g_alias_list = datum.get("G_alias_list", [])
            title, article = p_to_title_article[p]

            # skip papers with super long titles, which are the proceedings
            if len(title) > config.ore_max_title_tokens * 4:
                logger.info(f"title too long, skip: [#{prompt_id}]")
                continue

            # limit article size
            article = article[:config.ore_max_article_tokens * 4]

            # create prompt
            init_task_datum = DeepInfraTaskDatum(
                prompt_id,
                d, g, p, title, article,
                d_name, g_name, d_alias_list, g_alias_list,
                template,
            )
            logger.info(f"init: {init_task_datum.log_string}")

            # wait until quota is enough
            while requests_quota < 1:
                # let tasks run
                await asyncio.sleep(0.001)

                # process completed tasks
                new_task_to_datum = {}
                for running_task, running_task_datum in task_to_datum.items():
                    if running_task.done():
                        successful = False
                        try:
                            _running_task_datum = running_task.result()
                            successful = True
                            logger.info(f"done: {running_task_datum.log_string}")
                        except:
                            if running_task_datum.runs < config.ore_max_prompt_runs:
                                running_task = asyncio.create_task(
                                    ore_request(
                                        client, running_task_datum,
                                    )
                                )
                                new_task_to_datum[running_task] = running_task_datum
                                logger.info(f"re-run #{running_task_datum.runs}: {running_task_datum.log_string}")
                                await asyncio.sleep(0.0001)
                                continue
                            else:
                                running_task_datum.request_end_time = time.time()
                                logger.info(f"error: {running_task_datum.log_string}")

                        # save results
                        if successful:
                            json.dump(running_task_datum.get_json_obj(), fw)
                            fw.write("\n")
                            fw.flush()

                        heapq.heappush(
                            done_task_datum_queue,
                            (
                                running_task_datum.request_end_time,
                                done_task_datum_queue_next_id,
                                running_task_datum,
                            ),
                        )
                        done_task_datum_queue_next_id += 1

                    else:
                        new_task_to_datum[running_task] = running_task_datum

                task_to_datum = new_task_to_datum

                # process quota: reclaim quota from tasks
                while done_task_datum_queue:
                    heapq.heappop(done_task_datum_queue)
                    requests_quota += 1

            # deduct quota
            requests_quota -= 1

            # create a task and wait long enough so that request has been sent to server
            init_task = asyncio.create_task(
                ore_request(client, init_task_datum)
            )
            task_to_datum[init_task] = init_task_datum
            logger.info(f"run: {init_task_datum.log_string}")
            await asyncio.sleep(0.0001)

        # wait until all done
        while task_to_datum:
            done_task_set, pending_task_set = await asyncio.wait(task_to_datum, return_when=asyncio.FIRST_COMPLETED)
            new_task_to_datum = {
                pending_task: task_to_datum[pending_task]
                for pending_task in pending_task_set
            }

            for done_task in done_task_set:
                done_task_datum = task_to_datum[done_task]
                try:
                    _done_task_datum = done_task.result()
                    logger.info(f"done: {done_task_datum.log_string}")
                except:
                    if done_task_datum.runs < config.ore_max_prompt_runs:
                        done_task = asyncio.create_task(
                            ore_request(client, done_task_datum)
                        )
                        new_task_to_datum[done_task] = done_task_datum
                        logger.info(f"re-run #{done_task_datum.runs}: {done_task_datum.log_string}")
                        await asyncio.sleep(0.0001)
                    else:
                        done_task_datum.request_end_time = time.time()
                        logger.info(f"error: {done_task_datum.log_string}")
                    continue

                # save results
                json.dump(done_task_datum.get_json_obj(), fw)
                fw.write("\n")
                fw.flush()

            task_to_datum = new_task_to_datum

    logger.info("done")
    return


def get_truncated_text(text, tokenizer, max_tokens):
    tokens = len(tokenizer.encode(text))

    while tokens > max_tokens:
        cutoff_index = math.floor(len(text) * max_tokens / tokens) - 1
        if cutoff_index < 1:
            break
        text = text[:cutoff_index]
        tokens = len(tokenizer.encode(text))

    return text, tokens


async def run_ore_extraction_openai():
    from openai import AsyncOpenAI
    import tiktoken

    # set up client
    logger.info("setting up ORE client...")
    api_key = input("please input ORE server API key: ")
    logger.info("received server API key")
    client = AsyncOpenAI(
        api_key=api_key,
    )
    tokenizer = tiktoken.encoding_for_model(config.ore_model)

    # set up task management
    rpm_quota = config.ore_requests_per_minute
    tpm_quota = config.ore_tokens_per_minute
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # read prompt template
    with open(config.ore_prompt_file, "r", encoding="utf8") as f:
        template = f.read()

    # read completed data
    completed_prompt_id_set = set()
    if os.path.exists(config.ore_extraction_file):
        logger.info("reading completed data...")
        with open(config.ore_extraction_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_prompt_id_set.add(datum["prompt_id"])
        completed_prompts = len(completed_prompt_id_set)
        logger.info(f"read {completed_prompts:,} completed_prompts")

    # read article data
    logger.info("reading article data...")
    p_to_title_article = {}
    with open(config.literature_content_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            p = datum["P_id"]
            title = datum["title"]
            article = datum["article"]
            p_to_title_article[p] = (title, article)
    articles = len(p_to_title_article)
    logger.info(f"read {articles:,} articles")

    # run extraction by calling server
    logger.info("running extraction...")
    with open(config.literature_entity_file, "r", encoding="utf8") as fr, \
         open(config.ore_extraction_file, "a", encoding="utf8") as fw:

        for li, line in enumerate(fr):
            prompt_id = li + 1

            if prompt_id in completed_prompt_id_set:
                logger.info(f"skip: [#{prompt_id}]")
                continue

            # create task datum
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            p = datum["P_id"]
            d_name = datum["D_name"]
            g_name = datum["G_name"]
            d_alias_list = datum.get("D_alias_list", [])
            g_alias_list = datum.get("G_alias_list", [])
            title, article = p_to_title_article[p]

            # skip papers with super long titles, which are the proceedings
            title_tokens = len(tokenizer.encode(title))
            if title_tokens > config.ore_max_title_tokens * 4:
                logger.info(f"title too long, skip: [#{prompt_id}]")
                continue

            # limit article size
            article, _article_tokens = get_truncated_text(article, tokenizer, config.ore_max_article_tokens)

            # create prompt
            init_task_datum = OpenAITaskDatum(
                prompt_id,
                d, g, p, title, article,
                d_name, g_name, d_alias_list, g_alias_list,
                template, tokenizer,
            )
            logger.info(f"init: {init_task_datum.log_string}")

            # wait until quota is enough
            while rpm_quota < 1 or tpm_quota < init_task_datum.in_tokens:
                # let tasks run
                await asyncio.sleep(0.001)

                # process completed tasks
                new_task_to_datum = {}
                for running_task, running_task_datum in task_to_datum.items():
                    if running_task.done():
                        successful = False
                        try:
                            _running_task_datum = running_task.result()
                            successful = True
                            logger.info(f"done: {running_task_datum.log_string}")
                        except:
                            if running_task_datum.runs < config.ore_max_prompt_runs:
                                running_task = asyncio.create_task(
                                    ore_request(
                                        client, running_task_datum,
                                    )
                                )
                                new_task_to_datum[running_task] = running_task_datum
                                logger.info(f"re-run #{running_task_datum.runs}: {running_task_datum.log_string}")
                                await asyncio.sleep(0.0001)
                                continue
                            else:
                                running_task_datum.request_end_time = time.time()
                                logger.info(f"error: {running_task_datum.log_string}")

                        # save results
                        if successful:
                            json.dump(running_task_datum.get_json_obj(), fw)
                            fw.write("\n")
                            fw.flush()

                        heapq.heappush(
                            done_task_datum_queue,
                            (
                                running_task_datum.request_end_time,
                                done_task_datum_queue_next_id,
                                running_task_datum,
                            ),
                        )
                        done_task_datum_queue_next_id += 1

                    else:
                        new_task_to_datum[running_task] = running_task_datum

                task_to_datum = new_task_to_datum

                # process quota: reclaim quota from tasks
                while done_task_datum_queue:
                    request_end_time, _done_task_datum_queue_id, done_task_datum = done_task_datum_queue[0]
                    if request_end_time >= time.time() - 60:
                        break
                    heapq.heappop(done_task_datum_queue)
                    rpm_quota += 1
                    tpm_quota += done_task_datum.in_tokens

            # deduct quota
            rpm_quota -= 1
            tpm_quota -= init_task_datum.in_tokens

            # create a task and wait long enough so that request has been sent to server
            init_task = asyncio.create_task(
                ore_request(client, init_task_datum)
            )
            task_to_datum[init_task] = init_task_datum
            logger.info(f"run: {init_task_datum.log_string}")
            await asyncio.sleep(0.0001)

        # wait until all done
        while task_to_datum:
            done_task_set, pending_task_set = await asyncio.wait(task_to_datum, return_when=asyncio.FIRST_COMPLETED)
            new_task_to_datum = {
                pending_task: task_to_datum[pending_task]
                for pending_task in pending_task_set
            }

            for done_task in done_task_set:
                done_task_datum = task_to_datum[done_task]
                try:
                    _done_task_datum = done_task.result()
                    logger.info(f"done: {done_task_datum.log_string}")
                except:
                    if done_task_datum.runs < config.ore_max_prompt_runs:
                        done_task = asyncio.create_task(
                            ore_request(client, done_task_datum)
                        )
                        new_task_to_datum[done_task] = done_task_datum
                        logger.info(f"re-run #{done_task_datum.runs}: {done_task_datum.log_string}")
                        await asyncio.sleep(0.0001)
                    else:
                        done_task_datum.request_end_time = time.time()
                        logger.info(f"error: {done_task_datum.log_string}")
                    continue

                # save results
                json.dump(done_task_datum.get_json_obj(), fw)
                fw.write("\n")
                fw.flush()

            task_to_datum = new_task_to_datum

    logger.info("done")
    return


def run_ore_extraction():
    # check config
    global config

    assert os.path.exists(config.literature_entity_file)
    assert os.path.exists(config.literature_content_file)

    if config.ore_prompt_file:
        assert os.path.exists(config.ore_prompt_file)
    else:
        config.ore_prompt_file = "examples/LLM-ORE_prompt.txt"
        logger.info(f"<ore_prompt_file> is not specified. Will use {config.ore_prompt_file}.")

    if config.ore_server:
        # deepinfra
        if not config.ore_model:
            config.ore_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            logger.info(f"<ore_model> is not specified. Will use {config.ore_model}.")
        if config.ore_concurrent_requests <= 0:
            config.ore_concurrent_requests = 200
            logger.info(f"<ore_concurrent_requests> is not specified. Will use {config.ore_concurrent_requests}.")
            logger.info("check https://deepinfra.com/docs/advanced/rate-limits for more info")
        asyncio.run(run_ore_extraction_deepinfra())
    else:
        # openai
        logger.info("<ore_server> is not specified. Will use OpenAI server.")
        if not config.ore_model:
            config.ore_model = "gpt-4o-mini"
            logger.info(f"<ore_model> is not specified. Will use {config.ore_model}.")
        if config.ore_requests_per_minute <= 0:
            config.ore_requests_per_minute = 3  # gpt-4o-mini: [free tier: 3], [tier 5: 30,000]
            logger.info(
                f"<ore_requests_per_minute> is not specified. Will use {config.ore_requests_per_minute} (free tier)."
            )
            logger.info("check https://platform.openai.com/docs/guides/rate-limits for more info")
        if config.ore_tokens_per_minute <= 0:
            config.ore_tokens_per_minute = 40000  # gpt-4o-mini: [free tier: 40,000], [tier 5: 150,000,000]
            logger.info(
                f"<ore_tokens_per_minute> is not specified. Will use {config.ore_tokens_per_minute}. (free tier)"
            )
            logger.info("check https://platform.openai.com/docs/guides/rate-limits for more info")
        asyncio.run(run_ore_extraction_openai())
    return


def run_ore_knowledge_graph():
    assert os.path.exists(config.ore_extraction_file)

    header = ["D_id", "G_id", "P_id", "relation"]
    relation_pattern = re.compile(r'^- "([^"]+)", "([^"]+)", "([^"]+)"$')

    if config.ore_sort_knowledge_graph:
        logger.info("<ore_sort_knowledge_graph> is True, will store all relations in memory")
    else:
        logger.info("<ore_sort_knowledge_graph> is False, relations will not be sorted")
        fw = open(config.ore_knowledge_graph_file, "w", encoding="utf8", newline="")
        writer = csv.writer(fw, dialect="csv")
        writer.writerow(header)

    knowledge_graph = []
    all_relations = 0
    all_d_dict = {}
    all_g_dict = {}
    all_dg_dict = {}
    all_p_dict = {}

    logger.info("building knowledge graph...")
    with open(config.ore_extraction_file, "r", encoding="utf8") as fr:
        # each line corresponds to a prompt
        for line in fr:
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            p = datum["P_id"]
            d_name = datum["D_name"]
            g_name = datum["G_name"]
            d_alias_list = datum.get("D_alias_list", [])
            g_alias_list = datum.get("G_alias_list", [])
            text_out_list = datum["text_out_list"]

            accepted_relation_to_matcher = {}

            # each text_out corresponds to an LLM generation
            for text_out in text_out_list:
                relation_dict = {}

                # each text_out_line corresponds to a relation
                for text_out_line in text_out.split("\n"):
                    text_out_line = text_out_line.strip()
                    if not text_out_line.startswith("- "):
                        break
                    match = relation_pattern.fullmatch(text_out_line)
                    if match:
                        head, predicate, tail = match.group(1), match.group(2), match.group(3)

                        # check g in head
                        for alias in [g_name] + g_alias_list:
                            if alias in head:
                                break
                        else:
                            continue

                        # check d in tail
                        for alias in [d_name] + d_alias_list:
                            if alias in tail:
                                break
                        else:
                            continue

                        relation = f"{head} {predicate} {tail}"
                        relation_dict[relation] = True

                if not accepted_relation_to_matcher:
                    # this is the first successful generation
                    accepted_relation_to_matcher = {
                        relation: difflib.SequenceMatcher(b=relation, autojunk=False)
                        for relation in relation_dict
                    }

                else:
                    # add non-redundant relations from other successful generations
                    for candidate_relation in relation_dict:
                        for accepted_relation, matcher in accepted_relation_to_matcher.items():
                            if candidate_relation == accepted_relation:
                                break
                            matcher.set_seq1(candidate_relation)
                            if matcher.ratio() > config.ore_max_relation_similarity:
                                break
                        else:
                            accepted_relation_to_matcher[candidate_relation] = (
                                difflib.SequenceMatcher(b=candidate_relation, autojunk=False)
                            )

            if config.ore_sort_knowledge_graph:
                for relation in accepted_relation_to_matcher:
                    knowledge_graph.append([d, g, p, relation])
            else:
                for relation in accepted_relation_to_matcher:
                    writer.writerow([d, g, p, relation])
                    all_relations += 1

            if accepted_relation_to_matcher:
                all_d_dict[d] = True
                all_g_dict[g] = True
                all_dg_dict[(d, g)] = True
                all_p_dict[p] = True

    if config.ore_sort_knowledge_graph:
        all_relations = len(knowledge_graph)
    ds = len(all_d_dict)
    gs = len(all_g_dict)
    dgs = len(all_dg_dict)
    ps = len(all_p_dict)

    logger.info(
        f"built knowledge graph:"
        f" {ds:,} Ds; {gs:,} Gs; {dgs:,} DGs; {ps:,} Ps (articles); {all_relations:,} relations"
    )

    if config.ore_sort_knowledge_graph:
        logger.info("sorting knowledge graph...")
        knowledge_graph = sorted(knowledge_graph)
        with open(config.ore_knowledge_graph_file, "w", encoding="utf8", newline="") as fw:
            writer = csv.writer(fw, dialect="csv")
            writer.writerow(header)
            for row in knowledge_graph:
                writer.writerow(row)

    logger.info("done")
    return


def run_ore():
    run_ore_extraction()
    run_ore_knowledge_graph()
    return


"""
LLM-EMB
"""


def extract_dg_text():
    import tiktoken

    # collect text for each DGP
    logger.info("reading knowledge graph...")
    dg_p_text = defaultdict(lambda: defaultdict(lambda: []))
    dgp_dict = {}

    with open(config.ore_knowledge_graph_file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect="csv")
        header = next(reader)
        assert header == ["D_id", "G_id", "P_id", "relation"]
        for d, g, p, relation in reader:
            dg_p_text[(d, g)][p].append(f"{relation}.")
            dgp_dict[(d, g, p)] = True

    dg_p_text = {
        dg: {
            p: " ".join(text)
            for p, text in p_to_text.items()
        }
        for dg, p_to_text in dg_p_text.items()
    }

    dgs = len(dg_p_text)
    dgps = len(dgp_dict)
    del dgp_dict
    logger.info(f"read {dgs:,} DGs; {dgps:,} DGPs")

    # collect text list for each DG
    logger.info("creating text list for each DG...")
    tokenizer = tiktoken.encoding_for_model(config.emb_model)
    dg_to_text_tokens_list = {}
    texts = 0

    for dg, p_to_text in dg_p_text.items():
        block_text_tokens_list = []
        block_text = []
        block_tokens = 0

        for p, p_text in p_to_text.items():
            p_tokens = len(tokenizer.encode(p_text))
            assert p_tokens <= config.emb_max_text_tokens
            if block_text:
                if block_tokens + 1 + p_tokens > config.emb_max_text_tokens:
                    block_text = " ".join(block_text)
                    block_tokens = len(tokenizer.encode(block_text))
                    assert block_tokens <= config.emb_max_text_tokens
                    block_text_tokens_list.append((block_text, block_tokens))
                    texts += 1
                    block_text = [p_text]
                    block_tokens = p_tokens
                else:
                    block_text.append(p_text)
                    block_tokens += 1 + p_tokens
            else:
                block_text = [p_text]
                block_tokens = p_tokens

        if block_text:
            block_text = " ".join(block_text)
            block_tokens = len(tokenizer.encode(block_text))
            assert block_tokens <= config.emb_max_text_tokens
            block_text_tokens_list.append((block_text, block_tokens))
            texts += 1

        dg_to_text_tokens_list[dg] = block_text_tokens_list

    logger.info(f"created {texts:,} texts")

    return dg_to_text_tokens_list


class EmbTaskDatum:
    def __init__(self, prompt_id, d, g, text, tokens):
        self.prompt_id = prompt_id
        self.d = d
        self.g = g
        self.text = text
        self.tokens = tokens
        self.embedding = None

        self.runs = 0
        self.request_start_time = 0
        self.request_end_time = 0

        self.log_string = f"[#{self.prompt_id}] tokens={self.tokens:,} text[:100]={self.text[:100]}"
        return

    def get_json_obj(self):
        request_start_time = datetime.datetime.fromtimestamp(self.request_start_time).isoformat()
        request_end_time = datetime.datetime.fromtimestamp(self.request_end_time).isoformat()

        json_obj = {
            "prompt_id": self.prompt_id,
            "D_id": self.d, "G_id": self.g, "tokens": self.tokens,
            "runs": self.runs, "request_start_time": request_start_time, "request_end_time": request_end_time,
        }
        return json_obj


async def emb_request(async_client, task_datum):
    task_datum.runs += 1
    task_datum.request_start_time = time.time()
    completion = await async_client.embeddings.create(
        input=[task_datum.text],
        model=config.emb_model,
        dimensions=config.emb_dimension,
    )
    task_datum.request_end_time = time.time()

    task_datum.embedding = completion.data[0].embedding
    return task_datum


async def run_emb_extraction():
    from openai import AsyncOpenAI

    # check config
    assert os.path.exists(config.ore_knowledge_graph_file)
    if not config.emb_model:
        config.emb_model = "text-embedding-3-large"
        logger.info(f"<ore_model> is not specified. Will use {config.emb_model}.")
    if config.emb_requests_per_minute <= 0:
        config.emb_requests_per_minute = 100  # text-embedding-3-large: [free tier: 100], [tier 5: 30,000]
        logger.info(
            f"<emb_requests_per_minute> is not specified. Will use {config.emb_requests_per_minute}. (free tier)"
        )
        logger.info("check https://platform.openai.com/docs/guides/rate-limits for more info")
    if config.emb_tokens_per_minute <= 0:
        config.emb_tokens_per_minute = 10000  # text-embedding-3-large: [free tier: 10,000], [tier 5: 10,000,000]
        logger.info(
            f"<emb_tokens_per_minute> is not specified. Will use {config.emb_tokens_per_minute} (free tier)."
        )
        logger.info("check https://platform.openai.com/docs/guides/rate-limits for more info")

    # set up client
    logger.info("setting up ORE client...")
    api_key = input("please input EMB server API key: ")
    logger.info("received server API key")
    async_client = AsyncOpenAI(api_key=api_key)

    # set up task management
    rpm_quota = config.emb_requests_per_minute
    tpm_quota = config.emb_tokens_per_minute
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # read knowledge graph
    dg_to_text_tokens_list = extract_dg_text()

    # read completed data
    completed_prompt_id_set = set()
    if os.path.exists(config.emb_meta_file):
        logger.info("reading completed data...")
        with open(config.emb_meta_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_prompt_id_set.add(datum["prompt_id"])
        completed_prompts = len(completed_prompt_id_set)
        logger.info(f"read {completed_prompts:,} completed_prompts")

    # prompt embedding model for each text
    logger.info("running embedding...")
    with open(config.emb_meta_file, "a", encoding="utf8") as f_meta, \
         open(config.emb_bytes_file, "ab") as f_bytes:

        prompt_id = 0

        for (d, g), text_tokens_list in dg_to_text_tokens_list.items():
            for text, tokens in text_tokens_list:
                # create task
                prompt_id += 1
                if prompt_id in completed_prompt_id_set:
                    continue
                init_task_datum = EmbTaskDatum(prompt_id, d, g, text, tokens)
                logger.info(f"init: {init_task_datum.log_string}")

                # wait until quota is enough
                while rpm_quota < 1 or tpm_quota < init_task_datum.tokens:
                    # let tasks run
                    await asyncio.sleep(0.001)

                    # process completed tasks
                    new_task_to_datum = {}
                    for running_task, running_task_datum in task_to_datum.items():
                        if running_task.done():
                            successful = False
                            try:
                                _running_task_datum = running_task.result()
                                successful = True
                                logger.info(f"done: {running_task_datum.log_string}")
                            except:
                                if running_task_datum.runs < config.emb_max_prompt_runs:
                                    running_task = asyncio.create_task(
                                        emb_request(async_client, running_task_datum)
                                    )
                                    new_task_to_datum[running_task] = running_task_datum
                                    logger.info(f"re-run #{running_task_datum.runs}: {running_task_datum.log_string}")
                                    await asyncio.sleep(0.0001)
                                    continue
                                else:
                                    running_task_datum.request_end_time = time.time()
                                    logger.info(f"error: {running_task_datum.log_string}")

                            # save results
                            if successful:
                                json.dump(running_task_datum.get_json_obj(), f_meta)
                                f_meta.write("\n")
                                f_meta.flush()
                                for v in running_task_datum.embedding:
                                    v = struct.pack("d", v)
                                    f_bytes.write(v)

                            heapq.heappush(
                                done_task_datum_queue,
                                (
                                    running_task_datum.request_end_time,
                                    done_task_datum_queue_next_id,
                                    running_task_datum,
                                ),
                            )
                            done_task_datum_queue_next_id += 1

                        else:
                            new_task_to_datum[running_task] = running_task_datum

                    task_to_datum = new_task_to_datum

                    # process quota: reclaim quota from tasks finished over 1 minute
                    while done_task_datum_queue:
                        request_end_time, _done_task_datum_queue_id, done_task_datum = done_task_datum_queue[0]
                        if request_end_time >= time.time() - 60:
                            break
                        heapq.heappop(done_task_datum_queue)
                        rpm_quota += 1
                        tpm_quota += done_task_datum.tokens

                # deduct quota
                rpm_quota -= 1
                tpm_quota -= init_task_datum.tokens

                # create a task and wait long enough so that request has been sent to openai
                init_task = asyncio.create_task(
                    emb_request(async_client, init_task_datum)
                )
                task_to_datum[init_task] = init_task_datum
                logger.info(f"run: {init_task_datum.log_string}")
                await asyncio.sleep(0.0001)

        # wait until all done
        while task_to_datum:
            done_task_set, pending_task_set = await asyncio.wait(task_to_datum, return_when=asyncio.FIRST_COMPLETED)
            new_task_to_datum = {
                pending_task: task_to_datum[pending_task]
                for pending_task in pending_task_set
            }

            for done_task in done_task_set:
                done_task_datum = task_to_datum[done_task]
                try:
                    _done_task_datum = done_task.result()
                    logger.info(f"done: {done_task_datum.log_string}")
                except:
                    logger.info(traceback.format_exc())
                    if done_task_datum.runs < config.emb_max_prompt_runs:
                        done_task = asyncio.create_task(
                            emb_request(async_client, done_task_datum)
                        )
                        new_task_to_datum[done_task] = done_task_datum
                        logger.info(f"re-run #{done_task_datum.runs}: {done_task_datum.log_string}")
                        await asyncio.sleep(0.0001)
                    else:
                        done_task_datum.request_end_time = time.time()
                        logger.info(f"error: {done_task_datum.log_string}")
                    continue

                # save results
                json.dump(done_task_datum.get_json_obj(), f_meta)
                f_meta.write("\n")
                f_meta.flush()
                for v in done_task_datum.embedding:
                    v = struct.pack("d", v)
                    f_bytes.write(v)

            task_to_datum = new_task_to_datum

    logger.info("done")
    return


def run_emb_collection():
    logger.info(f"reading embedding bytes...")
    with open(config.emb_bytes_file, "rb") as f:
        emb_bytes = f.read()
    total_bytes = len(emb_bytes)
    logger.info(f"read {total_bytes:,} bytes")

    logger.info("collecting embedding...")
    dg_to_vector_tokens_list = defaultdict(lambda: [])
    vector_size = config.emb_dimension * 8  # a float is 8 bytes
    text_vectors = 0
    with open(config.emb_meta_file, "r", encoding="utf8") as f:
        bytes_i = 0

        for line in f:
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            tokens = datum["tokens"]

            bytes_j = bytes_i + vector_size
            vector = [v[0] for v in struct.iter_unpack("d", emb_bytes[bytes_i:bytes_j])]
            vector = np.array(vector, dtype=np.float64)
            bytes_i = bytes_j

            dg_to_vector_tokens_list[(d, g)].append((vector, tokens))
            text_vectors += 1
    dgs = len(dg_to_vector_tokens_list)
    logger.info(f"collected {text_vectors:,} vectors for {dgs:,} DGs")

    logger.info("creating embedding...")
    d_g_vector_data = []
    for (d, g), vector_tokens_list in dg_to_vector_tokens_list.items():
        merged_vector = np.zeros(config.emb_dimension, dtype=np.float64)
        weight = 0
        for vector, tokens in vector_tokens_list:
            merged_vector = merged_vector + vector * tokens
            weight += tokens
        merged_vector = merged_vector / weight
        d_g_vector_data.append((d, g, merged_vector))
    logger.info("created one vector for each DG")

    logger.info("saving...")
    with open(config.emb_file, "wb") as f:
        pickle.dump(d_g_vector_data, f)
    logger.info(f"done")
    return


def run_emb():
    global config

    if config.emb_dimension <= 0:
        config.emb_dimension = 256
        logger.info(f"<emb_dimension> is not specified. Will use {config.emb_dimension}.")

    asyncio.run(run_emb_extraction())
    run_emb_collection()
    return


"""
ML-Ranker
"""


def read_label_data():
    logger.info(f"reading label data...")
    d_g_label = defaultdict(lambda: {})
    g_set = set()

    with open(config.ranker_entity_label_file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect="csv")
        header = next(reader)
        assert header == ["D_id", "G_id", "label"]
        for d, g, label in reader:
            label = int(label)
            if label != 0:
                d_g_label[d][g] = label  # does not store 0 (negative label)
                g_set.add(g)

    ds = len(d_g_label)
    gs = len(g_set)
    dgs = sum(len(g_to_label) for g_to_label in d_g_label.values())
    logger.info(f"[DG label] {dgs:,} DGs: {ds:,} unique Ds, {gs:,} unique Gs")
    return d_g_label


def read_feature_data(d_g_label=None):
    logger.info(f"reading feature data...")
    d_g_feature = defaultdict(lambda: {})
    gold_d_dict = {}

    with open(config.ranker_entity_feature_file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect="csv")
        header = next(reader)
        assert header[:2] == ["D_id", "G_id"]
        for row in reader:
            d = row[0]
            g = row[1]

            # skip Ds that have no gold Gs
            if d_g_label is not None:
                if d not in d_g_label:
                    continue
                if d_g_label[d].get(g, 0) != 0:
                    gold_d_dict[d] = True

            feature = [float(i) for i in row[2:]]
            d_g_feature[d][g] = feature

    # skip Ds that have no gold Gs
    if d_g_label is not None:
        d_g_feature = {
            d: g_to_feature
            for d, g_to_feature in d_g_feature.items()
            if d in gold_d_dict
        }

    ds = len(d_g_feature)
    gs = len({g for g_to_feature in d_g_feature.values() for g in g_to_feature})
    dgs = sum(len(g_to_feature) for g_to_feature in d_g_feature.values())
    logger.info(f"[DG feature] {dgs:,} DGs: {ds:,} unique Ds, {gs:,} unique Gs")
    return d_g_feature, gold_d_dict


def read_embedding_data(gold_d_dict=None):
    logger.info("reading embedding data...")
    d_g_vector = defaultdict(lambda: {})
    g_set = set()

    with open(config.emb_file, "rb") as f:
        emb_data = pickle.load(f)

    for d, g, vector in emb_data:
        if gold_d_dict is not None and d not in gold_d_dict:  # skip Ds that have no gold Gs
            continue
        d_g_vector[d][g] = vector
        g_set.add(g)

    ds = len(d_g_vector)
    gs = len(g_set)
    dgs = sum(len(g_to_vector) for g_to_vector in d_g_vector.values())
    logger.info(f"[DG embedding] {dgs:,} DGs: {ds:,} unique Ds, {gs:,} unique Gs")
    return d_g_vector


def max_min_normalization(a):
    m = a.min(axis=0)
    s = a.max(axis=0) - m
    for i in range(s.shape[0]):
        if s[i] == 0:
            s[i] = 1
    b = (a - m) / s
    return b


def read_ranker_cross_validation_data():
    d_g_label = read_label_data()
    d_g_feature, gold_d_dict = read_feature_data(d_g_label)
    d_g_vector = read_embedding_data(gold_d_dict)

    logger.info("creating cross-validation data...")
    data = []

    def add_g():
        g_list.append(g)
        y_array.append(y)
        x_array.append(x)
        e = g_to_vector.get(g, np.zeros(config.emb_dimension, dtype=np.float64))
        e_array.append(e)
        return

    for d, g_to_feature in d_g_feature.items():
        g_to_label = d_g_label.get(d, {})
        g_to_vector = d_g_vector.get(d, {})

        g_list = []
        y_array = []
        x_array = []
        e_array = []

        # first, add positive DGs (y != 0)
        #   this is because during lightgbm training, only first 10,000 Gs (for each D) will be used
        #   and positive samples are more valuable than negative samples
        for g, x in g_to_feature.items():
            y = g_to_label.get(g, 0)
            if y != 0:
                add_g()

        # then, add negative DGs (y == 0)
        for g, x in g_to_feature.items():
            y = g_to_label.get(g, 0)
            if y == 0:
                add_g()

        y_array = np.array(y_array, dtype=np.float32)
        x_array = np.array(x_array, dtype=np.float32)
        x2_array = max_min_normalization(x_array)
        x3_array = np.array(e_array, dtype=np.float32)
        x_array = np.concatenate((x_array, x2_array, x3_array), axis=1)

        data.append((d, g_list, y_array, x_array))

    ds = len(data)
    logger.info(f"created data for {ds:,} Ds")
    return data, d_g_label


def read_ranker_test_data():
    d_g_feature, _gold_d_dict = read_feature_data()
    d_g_vector = read_embedding_data()

    logger.info("creating test data...")
    data = []

    for d, g_to_feature in d_g_feature.items():
        g_to_vector = d_g_vector.get(d, {})
        g_list = []
        x_array = []
        e_array = []

        for g, x in g_to_feature.items():
            g_list.append(g)
            x_array.append(x)
            e = g_to_vector.get(g, np.zeros(config.emb_dimension, dtype=np.float64))
            e_array.append(e)

        x_array = np.array(x_array, dtype=np.float32)
        x2_array = max_min_normalization(x_array)
        x3_array = np.array(e_array, dtype=np.float32)
        x_array = np.concatenate((x_array, x2_array, x3_array), axis=1)

        data.append((d, g_list, x_array))

    ds = len(data)
    logger.info(f"created data for {ds:,} Ds")
    return data


def get_map_recall(d_g_score, d_g_label):
    from sklearn.metrics import average_precision_score

    # recall
    golds = 0
    predicted_golds = 0

    # mean average precision (MAP)
    mean_ap = 0  # the MAP of the Ds that have predicted scores for gold DGs
    total_ranked_lists = 0  # number of the Ds that have predicted scores for gold DGs

    for d, g_to_label in d_g_label.items():
        if d not in d_g_score:
            continue
        gold_list = []
        score_list = []

        for g, score in d_g_score.get(d, {}).items():
            if g in g_to_label:
                gold_list.append(1)
            else:
                gold_list.append(0)
            score_list.append(score)

        this_d_predicted_golds = sum(gold_list)
        if this_d_predicted_golds == 0:
            continue

        predicted_golds += this_d_predicted_golds
        golds += len(g_to_label)

        ap = average_precision_score(gold_list, score_list, pos_label=1)
        mean_ap += ap
        total_ranked_lists += 1

    mean_ap /= total_ranked_lists
    recall = predicted_golds / golds
    return mean_ap, recall


def run_ranker_cross_validation():
    import lightgbm as lgb

    # check config
    assert os.path.exists(config.ranker_entity_feature_file)
    assert os.path.exists(config.ranker_entity_label_file)
    if config.ranker_D_split_file:
        assert os.path.exists(config.ranker_D_split_file)
    else:
        logger.info("<ranker_D_split_file> is not specified. Will do leave-one-D-out cross-validation.")
    assert os.path.exists(config.emb_file)
    assert config.emb_dimension > 0

    # read data
    data, d_g_label = read_ranker_cross_validation_data()

    # read cross-validation fold split
    logger.info(f"reading cross-validation folds...")
    if config.ranker_D_split_file:
        fold_to_d_list = defaultdict(lambda: [])
        with open(config.ranker_D_split_file, "r", encoding="utf8", newline="") as f:
            reader = csv.reader(f, dialect="csv")
            header = next(reader)
            assert header == ["D_id", "fold_id"]
            for d, fold in reader:
                fold_to_d_list[fold].append(d)
        fold_list = list(fold_to_d_list.values())
        del fold_to_d_list
        folds = len(fold_list)
        fold_size_list = [len(fold) for fold in fold_list]
        logger.info(f"{folds:,} fold cross-validation, #D-per-fold: {fold_size_list}")
        del fold_size_list
    else:
        fold_list = [
            [d]
            for d, _g_list, _y, _x in data
        ]
        folds = len(fold_list)
        logger.info(f"{folds:,} fold cross-validation, #D-per-fold: 1")

    # train and predict
    d_g_score = {}
    max_gs = 10000  # lightgbm limitation

    for fold_index, fold_d_list in enumerate(fold_list):
        fold_index += 1
        fold_d_set = set(fold_d_list)

        # training data
        y_train = []
        x_train = []
        q_train = []
        for split_d, _split_g_list, split_y, split_x in data:
            if split_d in fold_d_set:
                continue
            if split_y.shape[0] > max_gs:
                split_y = split_y[:max_gs]
                split_x = split_x[:max_gs]
            y_train.append(split_y)
            x_train.append(split_x)
            q_train.append(split_y.shape[0])
        y_train = np.concatenate(y_train, axis=0)
        x_train = np.concatenate(x_train, axis=0)

        # model training
        model = lgb.LGBMRanker(
            objective="lambdarank",
            device="cpu",
            verbosity=-1,  # log level: fatal
            force_row_wise=True,  # build histogram data-point-wise instead of feature-wise
            data_sample_strategy=config.ranker_lgb_data_sample_strategy,
            num_leaves=config.ranker_lgb_num_leaves,
            max_depth=config.ranker_lgb_max_depth,
        )
        model.fit(x_train, y_train, group=q_train)

        # testing data
        predicted_splits = 0
        for split_d, split_g_list, _split_y, split_x in data:
            if split_d not in fold_d_set:
                continue
            score_array = model.predict(split_x)
            d_g_score[split_d] = dict(zip(split_g_list, score_array))
            predicted_splits += 1
        logger.info(
            f"[fold #{fold_index:,}]"
            f" {len(q_train):,} trained Ds,"
            f" {predicted_splits:,} predicted Ds"
        )

    logger.info(f"saving predicted DG scores...")
    with open(config.ranker_entity_score_file, "wb") as f:
        pickle.dump(d_g_score, f)

    # evaluate
    logger.info(f"evaluating MAP...")
    mean_ap, recall = get_map_recall(d_g_score, d_g_label)
    logger.info(f"MAP={mean_ap: >5.1%} proportion_of_known_positive_DGs_predicted={recall: >5.1%}")

    logger.info("done")
    return


def run_ranker_train():
    import lightgbm as lgb

    # check config
    assert os.path.exists(config.ranker_entity_feature_file)
    assert os.path.exists(config.ranker_entity_label_file)
    assert os.path.exists(config.emb_file)
    assert config.emb_dimension > 0

    # read data
    data, d_g_label = read_ranker_cross_validation_data()

    # training data
    logger.info("creating training data...")
    max_gs = 10000  # lightgbm limitation
    y_train = []
    x_train = []
    q_train = []
    for _split_d, _split_g_list, split_y, split_x in data:
        if split_y.shape[0] > max_gs:
            split_y = split_y[:max_gs]
            split_x = split_x[:max_gs]
        y_train.append(split_y)
        x_train.append(split_x)
        q_train.append(split_y.shape[0])
    y_train = np.concatenate(y_train, axis=0)
    x_train = np.concatenate(x_train, axis=0)

    # model training
    logger.info("training model...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        device="cpu",
        verbosity=-1,  # log level: fatal
        force_row_wise=True,  # build histogram data-point-wise instead of feature-wise
        data_sample_strategy=config.ranker_lgb_data_sample_strategy,
        num_leaves=config.ranker_lgb_num_leaves,
        max_depth=config.ranker_lgb_max_depth,
    )
    model.fit(x_train, y_train, group=q_train)

    logger.info(f"saving model...")
    model.booster_.save_model(config.ranker_model_file)

    logger.info("done")
    return


def run_ranker_test():
    import lightgbm as lgb

    # check config
    assert os.path.exists(config.ranker_entity_feature_file)
    if config.ranker_entity_label_file:
        assert os.path.exists(config.ranker_entity_label_file)
    else:
        logger.info("<ranker_entity_label_file> is not specified. Will predict DG scores; will not evaluate.")
    assert os.path.exists(config.emb_file)
    assert config.emb_dimension > 0
    assert os.path.exists(config.ranker_model_file)

    # read data
    data = read_ranker_test_data()

    # read model
    logger.info(f"reading model...")
    model = lgb.Booster(model_file=config.ranker_model_file)

    # predict
    logger.info("predicting DG scores...")
    d_g_score = {}
    predicted_splits = 0
    for split_d, split_g_list, split_x in data:
        score_array = model.predict(split_x)
        d_g_score[split_d] = dict(zip(split_g_list, score_array))
        predicted_splits += 1
    logger.info(f"predicted for {predicted_splits:,} Ds")
    logger.info(f"saving predicted DG scores...")
    with open(config.ranker_entity_score_file, "wb") as f:
        pickle.dump(d_g_score, f)

    # evaluate
    if config.ranker_entity_label_file:
        d_g_label = read_label_data()
        logger.info(f"evaluating MAP...")
        mean_ap, recall = get_map_recall(d_g_score, d_g_label)
        logger.info(f"MAP={mean_ap: >5.1%} proportion_of_known_positive_DGs_predicted={recall: >5.1%}")

    logger.info("done")
    return


"""
Key-Semantics
"""


def extract_lemma_for_knowledge_graph():
    from nltk.tokenize.destructive import NLTKWordTokenizer
    from nltk.stem.snowball import SnowballStemmer

    # tokenize knowledge graph relations and extract lemmas
    logger.info("extracting lemmas for knowledge graph...")
    tokenizer = NLTKWordTokenizer()
    stemmer = SnowballStemmer("english")
    relations = 0
    lemma_to_relations = defaultdict(lambda: 0)

    with open(config.ore_knowledge_graph_file, "r", encoding="utf8", newline="") as fr, \
            open(config.semantics_lemma_file, "w", encoding="utf8") as fw:
        reader = csv.reader(fr, dialect="csv")

        header = next(reader)
        assert header == ["D_id", "G_id", "P_id", "relation"]

        for d, g, p, relation in reader:
            lemma_dict = {}
            for token in tokenizer.tokenize(relation):
                if len(token) <= 2:
                    continue
                lemma = stemmer.stem(token)
                lemma_dict[lemma] = True

            datum = {
                "D_id": d, "G_id": g, "P_id": p, "relation": relation,
                "lemma": list(lemma_dict.keys()),
            }
            json.dump(datum, fw)
            fw.write("\n")

            relations += 1
            for lemma in lemma_dict:
                lemma_to_relations[lemma] += 1

    lemmas = len(lemma_to_relations)
    logger.info(f"extracted {lemmas:,} lemmas for {relations:,} relations")
    return


def extract_candidate_key_semantics_lemma(d_g_label):
    # read knowledge graph relation lemmas and calculate stats
    logger.info("reading lemmas from knowledge graph...")
    lemma_to_dg_dict = defaultdict(lambda: {})
    lemma_label_relations = defaultdict(lambda: defaultdict(lambda: 0))

    with open(config.semantics_lemma_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            lemma_list = datum["lemma"]

            # we say that a lemma is inaccurate if it comes from some "negative" DG
            #   but when that D is simply not annotated at all, all its DG are not regarded as "negative"
            is_labeled_d = d in d_g_label
            label = d_g_label.get(d, {}).get(g, 0)

            for lemma in lemma_list:
                lemma_to_dg_dict[lemma][(d, g)] = True
                if is_labeled_d:
                    lemma_label_relations[lemma][label] += 1

    lemmas = len(lemma_to_dg_dict)
    logger.info(f"calculated stats for {lemmas:,} lemmas")

    # select high coverage and high precision lemmas as candidate key-semantics lemmas
    logger.info("extracting candidate key-semantics lemmas...")
    candidate_list = []
    for lemma, dg_dict in lemma_to_dg_dict.items():
        dgs = len(dg_dict)
        if dgs < config.semantics_min_DGs:
            continue
        label_to_relations = lemma_label_relations[lemma]
        try:
            gold_dg_relations = label_to_relations.get(1, 0) / sum(label_to_relations.values())
        except ZeroDivisionError:
            gold_dg_relations = 0
        if gold_dg_relations < config.semantics_min_gold_DG_relations:
            continue
        candidate_list.append((lemma, dgs, gold_dg_relations))
    candidate_list = sorted(candidate_list, key=lambda lemma_cov_acc: (-lemma_cov_acc[2], -lemma_cov_acc[1]))
    candidates = len(candidate_list)
    logger.info(f"extracted {candidates:,} candidate key-semantics lemmas")
    return candidate_list


def extract_lemma_relations_from_knowledge_graph(candidate_list, d_g_label):
    # collect relations for candidate key-semantics lemmas
    logger.info("reading relations for candidate key-semantics lemmas from knowledge graph...")
    lemma_label_d_g_relationlist = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))))
    candidate_lemma_dict = {
        lemma: True
        for lemma, dgs, gold_dg_relations in candidate_list
    }
    candidates = len(candidate_list)
    relations = 0
    with open(config.semantics_lemma_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            p = datum["P_id"]
            relation = datum["relation"]
            lemma_list = datum["lemma"]
            label = d_g_label.get(d, {}).get(g, 0)
            has_candidate_lemma = False

            for lemma in lemma_list:
                if lemma in candidate_lemma_dict:
                    lemma_label_d_g_relationlist[lemma][label][d][g].append(
                        (p, relation)
                    )
                    has_candidate_lemma = True

            if has_candidate_lemma:
                relations += 1
    logger.info(f"read {relations:,} relations for {candidates:,} candidate key-semantics lemmas")

    # sample a set of relations for candidate key-semantics lemmas
    logger.info(f"sampling {config.semantics_samples_per_lemma:,} relations for each key-semantics lemma label...")
    relations = 0
    with open(config.semantics_candidate_file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, dialect="csv")
        header = ["tag", "DGs", "gold DG relations", "D_id", "G_id", "DG label", "P_id", "relation"]
        writer.writerow(header)

        for lemma, dgs, gold_dg_relations in candidate_list:
            gold_dg_relations = f"{gold_dg_relations:.0%}"
            label_d_g_relationlist = lemma_label_d_g_relationlist.get(lemma, {})

            for label in sorted(label_d_g_relationlist):
                d_g_relationlist = label_d_g_relationlist[label]
                dg_list = [
                    (d, g)
                    for d, g_to_relationlist in d_g_relationlist.items()
                    for g in g_to_relationlist
                ]
                if len(dg_list) > config.semantics_samples_per_lemma:
                    dg_list = random.sample(dg_list, config.semantics_samples_per_lemma)
                for d, g in dg_list:
                    p, relation = random.choice(d_g_relationlist[d][g])
                    writer.writerow([lemma, dgs, gold_dg_relations, d, g, label, p, relation])
                    relations += 1
    logger.info(f"sampled {relations:,} relations for {candidates:,} candidate key-semantics lemmas")
    return


def read_semantics_data():
    tag_semantics_list = []

    logger.info("reading curated key semantics...")
    if config.semantics_file:
        with open(config.semantics_file, "r", encoding="utf8", newline="") as f:
            reader = csv.reader(f, dialect="csv")
            header = next(reader)
            assert header == ["tag", "semantics"]
            for lemma, semantics in reader:
                tag_semantics_list.append((lemma, semantics))

    else:
        semantics_file_list = os.listdir(config.semantics_taxonomy_path)
        for semantics_file in semantics_file_list:
            if not semantics_file.endswith(".txt"):
                continue
            semantics_file = os.path.join(config.semantics_taxonomy_path, semantics_file)
            with open(semantics_file, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("["):
                        continue
                    line = line.split(",")
                    if len(line) != 2:
                        continue
                    tag, semantics = line
                    tag_semantics_list.append((tag, semantics))

    tags = len(tag_semantics_list)
    logger.info(f"read {tags:,} key-semantics tags")
    return tag_semantics_list


def run_semantics_extraction():
    assert os.path.exists(config.ore_knowledge_graph_file)
    assert os.path.exists(config.ranker_entity_label_file)

    extract_lemma_for_knowledge_graph()
    d_g_label = read_label_data()
    candidate_list = extract_candidate_key_semantics_lemma(d_g_label)
    extract_lemma_relations_from_knowledge_graph(candidate_list, d_g_label)
    return


def run_semantics_tagging():
    assert os.path.exists(config.semantics_lemma_file)
    if config.semantics_file:
        logger.info("<semantics_file> is provided. <semantics_taxonomy_path> will not be used.")
        assert os.path.exists(config.semantics_file)
    else:
        logger.info("<semantics_file> is not provided. <semantics_taxonomy_path> will be used.")
        assert os.path.exists(config.semantics_taxonomy_path)

    all_tag_semantics_list = read_semantics_data()
    all_tag_to_order = {
        tag: index + 1
        for index, (tag, _semantics) in enumerate(all_tag_semantics_list)
    }

    logger.info("adding key semantics tags to knowledge graph...")
    header = ["D_id", "G_id", "P_id", "tag", "relation"]
    tagged_relations = 0
    relations = 0

    with open(config.semantics_lemma_file, "r", encoding="utf8") as fr, \
            open(config.semantics_knowledge_graph_file, "w", encoding="utf8", newline="") as fw:
        writer = csv.writer(fw, dialect="csv")
        writer.writerow(header)

        for line in fr:
            datum = json.loads(line)
            d = datum["D_id"]
            g = datum["G_id"]
            p = datum["P_id"]
            relation = datum["relation"]
            lemma_list = datum["lemma"]

            tag_list = []
            for lemma in lemma_list:
                order = all_tag_to_order.get(lemma, 0)
                if order > 0:
                    tag_list.append((order, lemma))

            relations += 1
            if tag_list:
                tagged_relations += 1

            tag_list = ",".join(lemma for _order, lemma in sorted(tag_list))
            writer.writerow([d, g, p, tag_list, relation])

    logger.info(
        f"built knowledge graph:"
        f" {tagged_relations:,} tagged relations;"
        f" {relations:,} all relations"
    )
    return


"""
Main
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[arg.{key}] {value}")

    global config
    config.load(arg.config_file)

    if config.task == "LLM-ORE":
        run_ore()

    elif config.task == "LLM-EMB":
        run_emb()

    elif config.task == "ML-Ranker_cross-validation":
        run_ranker_cross_validation()

    elif config.task == "ML-Ranker_train":
        run_ranker_train()

    elif config.task == "ML-Ranker_test":
        run_ranker_test()

    elif config.task == "Key-Semantics_extraction":
        run_semantics_extraction()

    elif config.task == "Key-Semantics_tagging":
        run_semantics_tagging()

    return


if __name__ == "__main__":
    main()
    sys.exit()
