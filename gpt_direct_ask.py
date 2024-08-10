import os
import json
import time
import heapq
import pickle
import asyncio
import logging
import datetime

from openai import AsyncOpenAI
import tiktoken

from llm_ore import gpt_request
from ml_ranker import get_map_recall

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


class DGPathoTaskDatum:
    def __init__(
            self, dgp_id,
            disease, gene, disease_name, gene_name,
            template, tokenizer,
    ):
        self.runs = 0
        self.dgp_id = dgp_id

        self.disease = disease
        self.gene = gene
        self.disease_name = disease_name
        self.gene_name = gene_name

        self.text_in = template.replace(
            "yoloGENEyolo", self.gene_name,
        ).replace(
            "yoloDISEASEyolo", self.disease_name,
        )
        self.text_out_list = []

        self.request_start_time = 0
        self.request_end_time = 0

        self.in_tokens = len(tokenizer.encode(self.text_in))
        self.out_tokens = 0

        self.log_string = (
            f"[#{self.dgp_id}]"
            f" [{self.disease}:{self.disease_name}]"
            f" [{self.gene}:{self.gene_name}]"
        )
        return

    def set_out_tokens(self, tokenizer):
        self.out_tokens = sum(
            len(tokenizer.encode(text_out))
            for text_out in self.text_out_list
        )
        return

    def get_json_obj(self):
        request_start_time = datetime.datetime.fromtimestamp(self.request_start_time).isoformat()
        request_end_time = datetime.datetime.fromtimestamp(self.request_end_time).isoformat()

        json_obj = {
            "dgp_id": self.dgp_id,
            "disease": self.disease, "gene": self.gene,
            "disease_name": self.disease_name, "gene_name": self.gene_name,
            "text_in": self.text_in,
            "text_out_list": self.text_out_list,
            "request_start_time": request_start_time, "request_end_time": request_end_time,
            "in_tokens": self.in_tokens, "out_tokens": self.out_tokens,
        }
        return json_obj


async def extract_gpt_pathogenicity_output_data(
        dg_data_file, gpt_output_file, gpt_choices, gpt_model, gpt_rpm, gpt_tpm, start, end,
):
    max_task_runs = 3

    # set up openai gpt
    api_key = input("API key: ")
    logger.info("received API key")
    client = AsyncOpenAI(api_key=api_key)
    tokenizer = tiktoken.encoding_for_model(gpt_model)

    # set up task management
    rpm_quota = gpt_rpm
    tpm_quota = gpt_tpm
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    template = 'Is the "yoloGENEyolo" gene pathogenic for the "yoloDISEASEyolo" disease? Answer in yes/no.'

    # read completed data
    completed_set = set()
    if os.path.exists(gpt_output_file):
        with open(gpt_output_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_set.add((datum["disease"], datum["gene"]))

    with open(dg_data_file, "r", encoding="utf8") as fr, \
         open(gpt_output_file, "a", encoding="utf8") as fw:

        for li, line in enumerate(fr):
            dgp_id = li + 1
            if dgp_id < start:
                continue
            if dgp_id > end:
                break

            # create task datum
            datum = json.loads(line)
            disease = datum["disease"]
            gene = datum["gene"]
            disease_name = datum["disease_name"]
            gene_name = datum["gene_name"]

            dg = (datum["disease"], gene)

            if dg in completed_set:
                logger.info(f"skip: [#{dgp_id}] [{disease}:{disease_name}] [{gene}:{gene_name}]")
            else:
                init_task_datum = DGPathoTaskDatum(
                    dgp_id,
                    disease, gene, disease_name, gene_name,
                    template, tokenizer,
                )
                logger.info(f"init: {init_task_datum.log_string}")

            # run tasks
            # wait until quota is enough
            while rpm_quota < 1 or tpm_quota < init_task_datum.in_tokens * 2:
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
                            if running_task_datum.runs < max_task_runs:
                                running_task = asyncio.create_task(
                                    gpt_request(client, gpt_model, gpt_choices, running_task_datum)
                                )
                                new_task_to_datum[running_task] = running_task_datum
                                logger.info(f"re-run #{running_task_datum.runs}: {running_task_datum.log_string}")
                                await asyncio.sleep(0.0001)
                                continue
                            else:
                                running_task_datum.request_end_time = time.time()
                                logger.info(f"error: {running_task_datum.log_string}")

                        # update tpm_quota with true out_tokens
                        running_task_datum.set_out_tokens(tokenizer)
                        tpm_quota = tpm_quota + running_task_datum.in_tokens - running_task_datum.out_tokens

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

                # process quota: reclaim quota from tasks finished over 1 minute
                while done_task_datum_queue:
                    request_end_time, _done_task_datum_queue_id, done_task_datum = done_task_datum_queue[0]
                    if request_end_time >= time.time() - 60:
                        break
                    heapq.heappop(done_task_datum_queue)
                    rpm_quota += 1
                    tpm_quota += done_task_datum.in_tokens + done_task_datum.out_tokens

            # deduct quota
            # use in_tokens * 2 when true in_tokens + out_tokens is unknown
            rpm_quota -= 1
            tpm_quota -= init_task_datum.in_tokens * 2

            # create a task and wait long enough so that request has been sent to openai
            init_task = asyncio.create_task(gpt_request(client, gpt_model, gpt_choices, init_task_datum))
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
                    if done_task_datum.runs < max_task_runs:
                        done_task = asyncio.create_task(
                            gpt_request(client, gpt_model, gpt_choices, done_task_datum)
                        )
                        new_task_to_datum[done_task] = done_task_datum
                        logger.info(f"re-run #{done_task_datum.runs}: {done_task_datum.log_string}")
                        await asyncio.sleep(0.0001)
                    else:
                        done_task_datum.request_end_time = time.time()
                        logger.info(f"error: {done_task_datum.log_string}")
                    continue

                # save results
                done_task_datum.set_out_tokens(tokenizer)
                json.dump(done_task_datum.get_json_obj(), fw)
                fw.write("\n")
                fw.flush()

            task_to_datum = new_task_to_datum

    logger.info("done")
    return


def extract_gpt_pathogenicity_prediction(gpt_output_file, prediction_file):
    method = "gpt-pathogenicity"
    logger.info(f"[{method}] reading GPT output from {gpt_output_file}")

    mesh_gene_prediction = {}
    answer_to_count = {"yes": 0, "no": 0, "other": 0}

    with open(gpt_output_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            mesh = datum["disease"]
            gene = datum["gene"]
            text_out_list = datum["text_out_list"]
            text_out = text_out_list[0].lower()

            if text_out.startswith("yes"):
                prediction = 1.0
                answer_to_count["yes"] += 1
            elif text_out.startswith("no"):
                prediction = 0.0
                answer_to_count["no"] += 1
            else:
                prediction = 0.5
                answer_to_count["other"] += 1

            gene_to_prediction = mesh_gene_prediction.get(mesh)
            if gene_to_prediction is None:
                gene_to_prediction = {}
                mesh_gene_prediction[mesh] = gene_to_prediction
            gene_to_prediction[gene] = prediction

    answers = sum(answer_to_count.values())
    for answer, count in answer_to_count.items():
        ratio = count / answers
        logger.info(f"[{answer}] {count:,}/{answers:,} = {ratio:.1%}")

    logger.info(f"[{method}] writing prediction to {prediction_file}")
    with open(prediction_file, "wb") as f:
        pickle.dump(mesh_gene_prediction, f)
    return


def evaluate_gpt_pathogenicity(prediction_file, db_pubmedkb_file):
    method = "gpt-pathogenicity"

    logger.info(f"[{method}] evaluating MAP and coverage...")
    mean_ap, recall = get_map_recall(db_pubmedkb_file, prediction_file)

    logger.info(f"[{method}] map={mean_ap: >5.1%} recall={recall: >5.1%}")
    return

