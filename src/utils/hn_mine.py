import argparse
import json
import random
import numpy as np
import faiss
from tqdm import tqdm
from ..model.bgem3 import M3ForInference
from arguments import ModelArguments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="./data/drugs_dataset.json", type=str)
    parser.add_argument('--output_file', default="./data/drugs_dataset_mine.json", type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_embedding', default=True, help='load model in gpu')
    parser.add_argument('--use_gpu_for_searching', default=False,
                        help='use faiss-gpu, faiss-gpu can only import on linux else faiss-cpu on win')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--embed_batch_size', default=4,
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


def find_knn_neg(
    model,
    input_file,
    output_file,
    sample_range,
    negative_number,
    use_gpu,
    embed_batch_size,
    search_batch_size,
):
    '''
    检索sample_range[-1]个最近的vec, 滤掉pos中的和与query相等的, 最终取negative_number个
    '''
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])
    corpus = list(set(corpus))

    p_vecs = model(corpus, batch_size=embed_batch_size)
    q_vecs = model(queries, batch_size=embed_batch_size)

    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1], batch_size=search_batch_size)
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus[inx] for inx in filtered_inx]

    with open(output_file, 'w') as f:
        for data in train_data:
            if len(data['neg']) < negative_number:
                samples = random.sample(corpus, negative_number - len(data['neg']) + len(data['pos']))
                samples = [sent for sent in samples if sent not in data['pos']]
                data['neg'].extend(samples[: negative_number - len(data['neg'])])
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    sample_range = list(map(int, args.range_for_sampling.split('-')))
    model = M3ForInference(model_load_args=ModelArguments(), device="cuda" if args.use_gpu_for_embedding else "cpu")

    find_knn_neg(model,
                 input_file=args.input_file,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 use_gpu=args.use_gpu_for_searching,
                 embed_batch_size=args.embed_batch_size,
                 search_batch_size=args.search_batch_size
                 )
