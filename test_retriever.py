from src.retriever.bing_retriever import BingRetriever
from transformers import HfArgumentParser
from src.utils.arguments import ModelArguments
import torch
import json
from tqdm import tqdm

with open("./data/mimic_drugs.json", "r") as f:
    all_durgs = json.load(f)

drug_desc = dict()

with torch.no_grad():
    parser = HfArgumentParser((ModelArguments, ))
    model_args = parser.parse_dict({"model_path":"./checkpoint/m3/"})[0]
    model_args: ModelArguments

    r = BingRetriever(model_args, device="cuda", scorer_max_batch_size=400)
    for drug in tqdm(all_durgs):
        drug_desc[drug] = r.query(f"What is {drug}?", result_length_high=4096, topk=5)

with open("./data/mimic_drug_desc.json", "w") as f:
    f.write(json.dumps(drug_desc))
