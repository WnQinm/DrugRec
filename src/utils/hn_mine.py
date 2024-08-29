import argparse
import json
import random
import numpy as np
import faiss
import torch
from tqdm import tqdm
from src.model.bgem3 import M3ForInference
from src.utils.arguments import ModelArguments
from typing import Dict, List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="./data/drugs_filtered.json", type=str)
    parser.add_argument('--output_file', default="./data/drugs_neg.json", type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_embedding', default=True, help='load model in gpu')
    parser.add_argument('--use_gpu_for_searching', default=False,
                        help='use faiss-gpu, faiss-gpu can only import on linux else faiss-cpu on win')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--embed_batch_size', default=1,
                        help="batch size when getting query/corpus embedding, independent with search_batch_size")
    parser.add_argument('--search_batch_size', default=64, help="search batch size with baiss knn")

    return parser.parse_args()


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index: faiss.IndexFlatIP,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


# TODO query和passage分开取负样本
def find_knn_neg(
    model,
    input_file,
    output_file,
    sample_range,
    negative_number,
    use_gpu,
    search_batch_size,
):

    train_data:Dict[str, List[str]] = dict()
    with open(input_file, "r") as f:
        input_data:Dict[str, dict] = json.load(f)
        drug_ids = list(input_data.keys())
        corpus:List[str] = [d["description"] for d in input_data.values()]
        # TODO random choice
        queries:List[str] = [d["names"][0] for d in input_data.values()]

    p_vecs = model(corpus).detach().cpu()
    q_vecs = model(queries).detach().cpu()

    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1], batch_size=search_batch_size)
    assert len(all_inxs) == len(drug_ids)

    for i in range(len(drug_ids)):
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        inxs = [inx for inx in inxs if inx != i]

        if len(inxs) > negative_number:
            inxs = random.sample(inxs, negative_number)
        elif len(inxs) < negative_number:
            samples = random.sample(corpus, negative_number - len(inxs) + 1)
            samples = [sent for sent in samples if sent != i]
            inxs.extend(samples[: negative_number - len(inxs)])
        train_data[drug_ids[i]] = [drug_ids[inx] for inx in inxs]

    with open(output_file, 'w') as f:
        f.write(json.dumps(train_data))


if __name__ == '__main__':
    with torch.no_grad():
        args = get_args()
        sample_range = list(map(int, args.range_for_sampling.split('-')))
        model = M3ForInference(model_load_args=ModelArguments(encode_sub_batch_size=args.embed_batch_size),
                               device="cuda" if args.use_gpu_for_embedding else "cpu")

        find_knn_neg(model,
                    input_file=args.input_file,
                    output_file=args.output_file,
                    sample_range=sample_range,
                    negative_number=args.negative_number,
                    use_gpu=args.use_gpu_for_searching,
                    search_batch_size=args.search_batch_size
                    )
