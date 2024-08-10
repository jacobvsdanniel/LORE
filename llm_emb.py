import os
import json
import time
import heapq
import pickle
import struct
import asyncio
import logging
import datetime
import traceback

import numpy as np
import tiktoken
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


class TaskDatum:
    def __init__(self, _id, text, tokenizer=None):
        self.id = _id
        self.text = text

        if tokenizer:
            self.tokens = len(tokenizer.encode(self.text))
        else:
            self.tokens = 0
        self.embedding = None

        self.runs = 0
        self.request_start_time = 0
        self.request_end_time = 0

        self.log_string = f"[#{self.id}] tokens={self.tokens:,} text[:100]={self.text[:100]}"
        return

    def get_json_obj(self):
        request_start_time = datetime.datetime.fromtimestamp(self.request_start_time).isoformat()
        request_end_time = datetime.datetime.fromtimestamp(self.request_end_time).isoformat()

        json_obj = {
            "id": self.id, "text": self.text, "tokens": self.tokens,
            "runs": self.runs, "request_start_time": request_start_time, "request_end_time": request_end_time,
        }
        return json_obj


async def async_embedding_request(async_client, model, task_datum):
    task_datum.runs += 1
    task_datum.request_start_time = time.time()
    completion = await async_client.embeddings.create(
        model=model,
        input=[task_datum.text]
    )
    task_datum.request_end_time = time.time()

    task_datum.embedding = completion.data[0].embedding
    return task_datum


async def async_extract_text_embedding(
        text_file, meta_file, embedding_file, openai_rpm, openai_tpm, start, end,
):
    max_task_runs = 10

    # file name
    meta_file = f"{meta_file}_{start}_{end}"
    embedding_file = f"{embedding_file}_{start}_{end}"

    # set up openai client
    api_key = input("API key: ")
    logger.info("received API key")
    async_client = AsyncOpenAI(api_key=api_key)
    model = "text-embedding-3-large"  # embedding size = 3072 double floats
    tokenizer = tiktoken.encoding_for_model(model)

    # set up task management
    rpm_quota = openai_rpm
    tpm_quota = openai_tpm
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # read completed data
    completed_set = set()
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_set.add(datum["id"])

    with open(text_file, "r", encoding="utf8") as f_text, \
         open(meta_file, "a", encoding="utf8") as f_meta, \
         open(embedding_file, "ab") as f_embedding:

        for li, line in enumerate(f_text):
            # create task
            text_id = li + 1
            if text_id < start:
                continue
            if text_id in completed_set:
                continue
            if text_id > end:
                break
            text = json.loads(line)
            init_task_datum = TaskDatum(text_id, text, tokenizer)
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
                            if running_task_datum.runs < max_task_runs:
                                running_task = asyncio.create_task(
                                    async_embedding_request(async_client, model, running_task_datum)
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
                                f_embedding.write(v)

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
                async_embedding_request(async_client, model, init_task_datum)
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
                    if done_task_datum.runs < max_task_runs:
                        done_task = asyncio.create_task(
                            async_embedding_request(async_client, model, done_task_datum)
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
                    f_embedding.write(v)

            task_to_datum = new_task_to_datum

    logger.info("done")
    return


def extract_reduced_embedding(src_file, tgt_file, src_dim, tgt_dim):
    # float -> 8 bytes
    src_vector_size = src_dim * 8
    tgt_vector_size = tgt_dim * 8

    # process n vectors at a time
    block_n = 500
    block_size = block_n * src_vector_size

    vectors = 0

    with open(src_file, "rb") as fr, \
         open(tgt_file, "wb") as fw:
        while True:
            bytes_data = fr.read(block_size)
            if not bytes_data:
                break
            read_size = len(bytes_data)
            read_n = int(read_size / src_vector_size)
            vectors += read_n

            data = []
            for ni in range(read_n):
                vi = ni * src_vector_size
                vj = vi + tgt_vector_size
                data.append([
                    v[0]
                    for v in struct.iter_unpack("d", bytes_data[vi:vj])
                ])
            data = np.array(data, dtype=np.float64)

            norm = np.linalg.norm(data, 2, axis=1, keepdims=True)
            data = np.where(norm == 0, data, data / norm)

            for vector in data:
                for v in vector:
                    v = struct.pack("d", v)
                    fw.write(v)

            if vectors % 100000 == 0:
                logger.info(f"processed {vectors:,} vectors")
        if vectors % 100000 != 0:
            logger.info(f"processed {vectors:,} vectors")
    return


def extract_dg_embedding_id_file(embedding_meta_file, dg_text_file, dg_embedding_id_file):
    # create mapping of text to embedding ID
    text_to_id = {}
    with open(embedding_meta_file, "r", encoding="utf8") as f:
        for li, line in enumerate(f):
            data = json.loads(line)
            text = data["text"]
            text_to_id[text] = li

            li += 1
            if li % 1000000 == 0:
                logger.info(f"processed {li:,} texts")
        if li % 1000000 != 0:
            logger.info(f"processed {li:,} texts")

    with open(dg_text_file, "r", encoding="utf8") as fr, \
         open(dg_embedding_id_file, "w", encoding="utf8") as fw:
        dgs = 0
        for line in fr:
            dgs += 1
            d, g, triplet_data, paper_data, text_data = json.loads(line)

            text_data = [triplet_data, paper_data, text_data]
            embedding_id_data = [
                [
                    (text_to_id[text], tokens)
                    for text, tokens in text_tokens_list
                ]
                for text_tokens_list in text_data
            ]

            embedding_id_data = [d, g, *embedding_id_data]
            json.dump(embedding_id_data, fw)
            fw.write("\n")

            if dgs % 10000 == 0:
                logger.info(f"processed {dgs:,} DGs")
        if dgs % 10000 != 0:
            logger.info(f"processed {dgs:,} DGs")

    return


def extract_dg_embedding(dg_embedding_id_file, embedding_file, embedding_dimension, dg_embedding_file):
    logger.info(f"reading {embedding_file}")
    with open(embedding_file, "rb") as f:
        embedding_bytes = f.read()
    logger.info("done reading bytes")

    vector_size = embedding_dimension * 8  # a float is 8 bytes

    def get_combined_vector(embedding_id_tokens_list, weighted_by_tokens):
        vector_sum = np.zeros(embedding_dimension, dtype=np.float64)
        weight_sum = 0

        for embedding_id, tokens in embedding_id_tokens_list:
            i = embedding_id * vector_size
            j = i + vector_size

            vector = [v[0] for v in struct.iter_unpack("d", embedding_bytes[i:j])]
            vector = np.array(vector, dtype=np.float64)

            weight = tokens if weighted_by_tokens else 1
            vector_sum = vector_sum + vector * weight
            weight_sum += weight

        return vector_sum / weight_sum

    def get_one_vector(embedding_id_tokens_list):
        embedding_id = embedding_id_tokens_list[0][0]
        i = embedding_id * vector_size
        j = i + vector_size
        vector = [v[0] for v in struct.iter_unpack("d", embedding_bytes[i:j])]
        vector = np.array(vector, dtype=np.float64)
        return vector

    logger.info(f"processing {dg_embedding_id_file}")
    dg_embedding_data = []

    with open(dg_embedding_id_file, "r", encoding="utf8") as f:
        dgs = 0
        triplets = 0
        log_triplets = 0

        for line in f:
            dgs += 1
            d, g, triplet_data, paper_data, text_data = json.loads(line)
            triplets += len(triplet_data)
            log_triplets += len(triplet_data)

            triplet_vector = get_combined_vector(triplet_data, False)
            paper_vector = get_combined_vector(paper_data, True)
            text_vector = get_combined_vector(text_data, True)
            one_vector = get_one_vector(text_data)

            type_to_vector = {
                "triplet": triplet_vector,
                "paper": paper_vector,
                "text": text_vector,
                "one": one_vector,
            }
            dg_embedding_data.append((d, g, type_to_vector))

            if log_triplets >= 100000:
                logger.info(f"processed: {dgs:,} DGs; {triplets:,} triplets")
                log_triplets -= 100000
        logger.info(f"finished: {dgs:,} DGs; {triplets:,} triplets")

    with open(dg_embedding_file, "wb") as f:
        pickle.dump(dg_embedding_data, f)
    logger.info(f"saved to {dg_embedding_file}")
    return

