import os
import re
import json
import math
import time
import heapq
import asyncio
import difflib
import logging
import datetime
from collections import defaultdict

from openai import AsyncOpenAI
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


class TaskDatum:
    def __init__(
            self, dgp_id,
            mesh, gene, pmid, title, abstract,
            head, tail,
            template, tokenizer,
    ):
        self.runs = 0
        self.dgp_id = dgp_id

        self.mesh = mesh
        self.gene = gene
        self.pmid = pmid
        self.title = title
        self.abstract = abstract

        self.head = head
        self.tail = tail

        self.text_in = template.replace(
            "yoloTITLEyolo", self.title,
        ).replace(
            "yoloTEXTyolo", self.abstract,
        ).replace(
            "yoloGENEyolo", self.head,
        ).replace(
            "yoloDISEASEyolo", self.tail,
        )
        self.text_out_list = []

        self.request_start_time = 0
        self.request_end_time = 0

        self.in_tokens = len(tokenizer.encode(self.text_in))
        self.out_tokens = 0

        self.log_string = (
            f"[#{self.dgp_id}]"
            f" [{self.mesh}]"
            f" [GENE:{self.gene}]"
            f" [PMID:{self.pmid}]"
            f" [HEAD:{self.head}]"
            f" [TAIL:{self.tail}]"
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
            "mesh": self.mesh, "gene": self.gene, "pmid": self.pmid,
            "head": self.head, "tail": self.tail,
            "text_out_list": self.text_out_list,
            "request_start_time": request_start_time, "request_end_time": request_end_time,
            "in_tokens": self.in_tokens, "out_tokens": self.out_tokens,
        }
        return json_obj


async def gpt_request(client, model, choices, task_datum):
    task_datum.runs += 1
    task_datum.request_start_time = time.time()
    completion = await client.chat.completions.create(
        model=model,
        n=choices,
        messages=[
            {"role": "user", "content": task_datum.text_in},
        ]
    )
    task_datum.request_end_time = time.time()

    task_datum.text_out_list = [
        choice.message.content
        for choice in completion.choices
    ]
    return task_datum


def get_truncated_text(text, tokenizer, max_tokens):
    tokens = len(tokenizer.encode(text))

    while tokens > max_tokens:
        cutoff_index = math.floor(len(text) * max_tokens / tokens) - 1
        if cutoff_index < 1:
            break
        text = text[:cutoff_index]
        tokens = len(tokenizer.encode(text))

    return text, tokens


async def extract_gpt_triplet_output_data(
        name_property_file, gpt_output_file, prompt_dir, gpt_choices, gpt_rpm, gpt_tpm, start, end,
):
    max_title_tokens = 200
    max_abstract_tokens = 2500
    max_task_runs = 10

    # set up openai gpt
    api_key = input("API key: ")
    logger.info("received API key")
    client = AsyncOpenAI(api_key=api_key)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # set up task management
    rpm_quota = gpt_rpm
    tpm_quota = gpt_tpm
    task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # read gpt prompt template
    prompt_file = os.path.join(prompt_dir, "extract_triplet.txt")
    with open(prompt_file, "r", encoding="utf8") as f:
        template = f.read()

    # read completed data
    completed_set = set()
    if os.path.exists(gpt_output_file):
        with open(gpt_output_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_set.add((datum["dgp_id"], datum["head"]))

    with open(name_property_file, "r", encoding="utf8") as fr, \
         open(gpt_output_file, "a", encoding="utf8") as fw:

        for li, line in enumerate(fr):
            dgp_id = li + 1
            if dgp_id < start:
                continue
            if dgp_id > end:
                break

            # create task datum
            datum = json.loads(line)
            mesh = datum["mesh"]
            gene = datum["gene"]
            pmid = datum["pmid"]
            mesh_gpt_name = datum["mesh_gpt_name"]
            gene_gpt_name = datum["gene_gpt_name"]
            variant_gpt_name_list = datum["variant_gpt_name_list"]
            title = datum["title"]
            abstract = datum["abstract"]

            # skip papers with super long titles, which are the proceedings
            title_tokens = len(tokenizer.encode(title))
            if title_tokens > max_title_tokens:
                logger.info(f"title_too_long: skip: [#{dgp_id}]")
                continue

            # limit abstract size
            abstract, _abstract_tokens = get_truncated_text(abstract, tokenizer, max_abstract_tokens)

            # create gpt input
            if gene_gpt_name and gene_gpt_name not in variant_gpt_name_list:
                head_list = [gene_gpt_name] + variant_gpt_name_list
            else:
                head_list = variant_gpt_name_list

            init_task_datum_list = []

            for head in head_list:
                if (dgp_id, head) in completed_set:
                    logger.info(f"skip: [#{dgp_id}] [{head}]")
                else:
                    init_task_datum = TaskDatum(
                        dgp_id,
                        mesh, gene, pmid, title, abstract,
                        head, mesh_gpt_name,
                        template, tokenizer,
                    )
                    init_task_datum_list.append(init_task_datum)
                    logger.info(f"init: {init_task_datum.log_string}")

            # run tasks
            for init_task_datum in init_task_datum_list:
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
                                        gpt_request(client, "gpt-3.5-turbo", gpt_choices, running_task_datum)
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
                init_task = asyncio.create_task(gpt_request(client, "gpt-3.5-turbo", gpt_choices, init_task_datum))
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
                            gpt_request(client, "gpt-3.5-turbo", gpt_choices, done_task_datum)
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


def extract_gpt_triplet_raw_triplet_data(name_property_file, gpt_output_file, gpt_raw_triplet_file, start, end):
    dgpid_head_gptdatum = defaultdict(lambda: {})

    with open(gpt_output_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            dgp_id = datum["dgp_id"]
            if start <= dgp_id <= end:
                head = datum["head"]
                dgpid_head_gptdatum[dgp_id][head] = datum

    gpt_output_dgps = len(dgpid_head_gptdatum)
    gpt_output_heads = sum(len(head_to_gptdatum) for dgpid, head_to_gptdatum in dgpid_head_gptdatum.items())
    logger.info(f"[gpt_output] {gpt_output_dgps:,} dgps; {gpt_output_heads:,} heads")

    dgp_id = 0
    gpt_collect_dgps = 0
    gpt_collect_heads = 0

    with open(name_property_file, "r", encoding="utf8") as fr, \
         open(gpt_raw_triplet_file, "w", encoding="utf8") as fw:
        for line in fr:
            dgp_id += 1

            if dgp_id in dgpid_head_gptdatum:
                gpt_collect_dgps += 1
                gpt_collect_heads += len(dgpid_head_gptdatum[dgp_id])

                datum = json.loads(line)
                datum["gpt_head_output_list"] = []

                for head, gpt_out_datum in dgpid_head_gptdatum[dgp_id].items():
                    datum["gpt_head_output_list"].append(
                        (head, gpt_out_datum["text_out_list"])
                    )

                json.dump(datum, fw)
                fw.write("\n")

            if dgp_id % 100000 == 0:
                logger.info(
                    f"[gpt_collect]"
                    f" {dgp_id:,} all_target_dgps;"
                    f" {gpt_collect_dgps:,} collected_dgps;"
                    f" {gpt_collect_heads:,} collected_heads"
                )

        if dgp_id % 100000 != 0:
            logger.info(
                f"[gpt_collect]"
                f" {dgp_id:,} all_target_dgps;"
                f" {gpt_collect_dgps:,} collected_dgps;"
                f" {gpt_collect_heads:,} collected_heads"
            )
    return


def extract_gpt_triplet_parse_data(gpt_raw_file, gpt_parse_file):
    pattern = re.compile(r'^- "([^"]+)", "([^"]+)", "([^"]+)"$')
    dgps = 0

    with open(gpt_raw_file, "r", encoding="utf8") as fr, \
         open(gpt_parse_file, "w", encoding="utf8") as fw:
        for line in fr:
            datum = json.loads(line)
            gpt_head_output_list = datum["gpt_head_output_list"]
            gpt_head_parse_list = []

            for head, raw_text_list in gpt_head_output_list:
                parse = []

                # parse each gpt text output to a triplet list
                for text in raw_text_list:
                    triplet_set = set()
                    text = "- " + text

                    # processing each output line
                    for text_line in text.split("\n"):
                        text_line = text_line.strip()

                        # well formatted output ends
                        if not text_line.startswith("- "):
                            break

                        match = pattern.fullmatch(text_line)
                        if match:
                            triplet = (match.group(1), match.group(2), match.group(3))
                            triplet_set.add(triplet)

                    parse.append(list(triplet_set))

                gpt_head_parse_list.append((head, parse))

            datum["gpt_head_parse_list"] = gpt_head_parse_list
            json.dump(datum, fw)
            fw.write("\n")

            dgps += 1
            if dgps % 100000 == 0:
                logger.info(f"{dgps:,} dgps")
        if dgps % 100000 != 0:
            logger.info(f"{dgps:,} dgps")
    return


def extract_gpt_triplet_match_data(gpt_triplet_parse_file, gpt_triplet_match_file):
    triplets = 0
    good_triplets = 0
    dgps = 0

    with open(gpt_triplet_parse_file, "r", encoding="utf8") as fr, \
         open(gpt_triplet_match_file, "w", encoding="utf8") as fw:
        for line in fr:
            datum = json.loads(line)
            mesh = datum["mesh"]
            gene = datum["gene"]
            pmid = datum["pmid"]
            mesh_name_to_stats = datum["mesh_name_to_stats"]
            gene_name_to_stats = datum["gene_name_to_stats"]
            vid_name_count = datum["vid_name_count"]
            name_vid_count = datum["name_vid_count"]
            gpt_head_parse_list = datum["gpt_head_parse_list"]

            mesh_name_to_matches = defaultdict(lambda: 0)
            gene_name_to_matches = defaultdict(lambda: 0)
            vid_name_matches = defaultdict(lambda: defaultdict(lambda: 0))

            gpt_match_extraction_list = []
            gpt_other_extraction_list = []

            for input_h, parse in gpt_head_parse_list:
                # match input to g/v ids
                input_gene = ""
                input_vid_list = []
                candidate_v_name_set = set()

                if input_h in gene_name_to_stats:
                    input_gene = gene

                for vid in name_vid_count.get(input_h, {}):
                    input_vid_list.append(vid)
                    for name in vid_name_count[vid]:
                        candidate_v_name_set.add(name)

                # match output triplet to gene/variant/disease
                match_output_list = []
                other_output_list = []

                for triplet_list in parse:
                    match_output = []
                    other_output = []

                    for triplet in triplet_list:
                        h, r, t = triplet
                        triplet_mesh_name_to_matches = defaultdict(lambda: 0)
                        triplet_gene_name_to_matches = defaultdict(lambda: 0)
                        triplet_vid_name_matches = defaultdict(lambda: defaultdict(lambda: 0))

                        for name in mesh_name_to_stats:
                            if name in t:
                                triplet_mesh_name_to_matches[name] += 1

                        if input_gene:
                            for name in gene_name_to_stats:
                                if name in h:
                                    triplet_gene_name_to_matches[name] += 1

                        for name in candidate_v_name_set:
                            if name in h:
                                for vid in name_vid_count[name]:
                                    triplet_vid_name_matches[vid][name] += 1

                        triplets += 1
                        if (triplet_gene_name_to_matches or triplet_vid_name_matches) and triplet_mesh_name_to_matches:
                            for name, matches in triplet_mesh_name_to_matches.items():
                                mesh_name_to_matches[name] += 1

                            for name, matches in triplet_gene_name_to_matches.items():
                                gene_name_to_matches[name] += 1

                            for vid, name_to_matches in triplet_vid_name_matches.items():
                                for name, matches in name_to_matches.items():
                                    vid_name_matches[vid][name] += matches

                            good_triplets += 1
                            match_output.append({
                                "triplet": triplet,
                                "mesh_name_to_matches": triplet_mesh_name_to_matches,
                                "gene_name_to_matches": triplet_gene_name_to_matches,
                                "vid_name_matches": triplet_vid_name_matches,
                            })
                        else:
                            other_output.append({
                                "triplet": triplet,
                                "mesh_name_to_matches": triplet_mesh_name_to_matches,
                                "gene_name_to_matches": triplet_gene_name_to_matches,
                                "vid_name_matches": triplet_vid_name_matches,
                            })

                    match_output_list.append(match_output)
                    other_output_list.append(other_output)

                match_extraction = {
                    "input_gene": input_gene,
                    "input_vid_list": input_vid_list,
                    "match_output_list": match_output_list,
                }
                other_extraction = {
                    "input_gene": input_gene,
                    "input_vid_list": input_vid_list,
                    "other_output_list": other_output_list,
                }

                gpt_match_extraction_list.append(match_extraction)
                gpt_other_extraction_list.append(other_extraction)

            extraction_datum = {
                "mesh": mesh, "gene": gene, "pmid": pmid,
                "mesh_name_to_matches": mesh_name_to_matches,
                "gene_name_to_matches": gene_name_to_matches,
                "vid_name_matches": vid_name_matches,
                "gpt_match_extraction_list": gpt_match_extraction_list,
                "gpt_other_extraction_list": gpt_other_extraction_list,
            }
            json.dump(extraction_datum, fw)
            fw.write("\n")

            dgps += 1
            if dgps % 100000 == 0:
                logger.info(f"{dgps:,} dgps: {good_triplets:,}/{triplets:,} good/all triplets")
        if dgps % 100000 != 0:
            logger.info(f"{dgps:,} dgps: {good_triplets:,}/{triplets:,} good/all triplets")
    return


def extract_gpt_triplet_combined_data(gpt_triplet_match_file, gpt_triplet_combine_file, max_similarity):
    has_triplet_dgps = 0
    dgps = 0

    combined_triplets = 0
    triplets = 0

    def get_flat_datum(_triplet_datum):
        return (
            *_triplet_datum["triplet"],
            _triplet_datum["mesh_name_to_matches"],
            _triplet_datum["gene_name_to_matches"],
            _triplet_datum["vid_name_matches"],
        )

    with open(gpt_triplet_match_file, "r", encoding="utf8") as fr, \
         open(gpt_triplet_combine_file, "w", encoding="utf8") as fw:
        for line in fr:
            datum = json.loads(line)
            mesh = datum["mesh"]
            gene = datum["gene"]
            pmid = datum["pmid"]

            mesh_name_to_matches = datum["mesh_name_to_matches"]
            gene_name_to_matches = datum["gene_name_to_matches"]
            vid_name_matches = datum["vid_name_matches"]

            gpt_match_extraction_list = datum["gpt_match_extraction_list"]
            gpt_combined_extraction_list = []

            for extraction in gpt_match_extraction_list:
                is_first_choice = True
                clause_to_matcher = {}

                for output in extraction["match_output_list"]:
                    if not output:
                        continue

                    if is_first_choice:
                        is_first_choice = False
                        for triplet_datum in output:
                            triplets += 1
                            gpt_combined_extraction_list.append(get_flat_datum(triplet_datum))
                            clause = " ".join(triplet_datum["triplet"])
                            clause_to_matcher[clause] = difflib.SequenceMatcher(b=clause, autojunk=False)

                    else:
                        for triplet_datum in output:
                            triplets += 1
                            clause = " ".join(triplet_datum["triplet"])
                            for other, matcher in clause_to_matcher.items():
                                if clause == other:
                                    break
                                matcher.set_seq1(clause)
                                if matcher.ratio() > max_similarity:
                                    break
                            else:
                                gpt_combined_extraction_list.append(get_flat_datum(triplet_datum))
                                clause_to_matcher[clause] = difflib.SequenceMatcher(b=clause, autojunk=False)

            # sort extraction
            index_list = list(range(len(gpt_combined_extraction_list)))
            index_to_key = [
                (-len(g), tuple(sorted(v.keys())))
                for h, r, t, m, g, v in gpt_combined_extraction_list
            ]
            index_list = sorted(index_list, key=lambda i: index_to_key[i])
            gpt_combined_extraction_list = [
                gpt_combined_extraction_list[index]
                for index in index_list
            ]

            if gpt_combined_extraction_list:
                has_triplet_dgps += 1
                combined_triplets += len(gpt_combined_extraction_list)
            dgps += 1

            combined_datum = {
                "mesh": mesh, "gene": gene, "pmid": pmid,
                "mesh_name_to_matches": mesh_name_to_matches,
                "gene_name_to_matches": gene_name_to_matches,
                "vid_name_matches": vid_name_matches,
                "gpt_extraction_list": gpt_combined_extraction_list,
            }
            json.dump(combined_datum, fw)
            fw.write("\n")

            if dgps % 100000 == 0:
                logger.info(f"#{dgps:,}")
                logger.info(f"{has_triplet_dgps:,}/{dgps:,} has-triplet/all DGPs")
                logger.info(f"{combined_triplets:,}/{triplets:,} combined/all triplets")
        if dgps % 100000 != 0:
            logger.info(f"#{dgps:,}")
            logger.info(f"{has_triplet_dgps:,}/{dgps:,} has-triplet/all DGPs")
            logger.info(f"{combined_triplets:,}/{triplets:,} combined/all triplets")
    return

