import torch
from transformers import AutoTokenizer, AutoModel
import os

from typing import List, Dict

class ContrieverScorer:
    def __init__(self, tokenizer_path, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
        query_encoder_path = os.path.join(retriever_ckpt_path, 'query_encoder')
        reference_encoder_path = os.path.join(retriever_ckpt_path, 'reference_encoder')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.query_encoder = AutoModel.from_pretrained(query_encoder_path).to(self.device).eval()
        self.reference_encoder = AutoModel.from_pretrained(reference_encoder_path).to(self.device).eval()

        assert max_batch_size > 0
        self.max_batch_size = max_batch_size

    def get_embeddings(self, sentences: List[str], mode: str) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True,
                                    truncation=True, return_tensors='pt')
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            if mode == "query":
                outputs = self.query_encoder(**inputs)
            elif mode == "reference":
                outputs = self.reference_encoder(**inputs)
            else:
                raise NotImplementedError

            # Mean Pool
            token_embeddings = outputs[0]
            mask = inputs["attention_mask"]
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

    def score_documents_on_query(self, query: str, documents: List[str]) -> torch.Tensor:
        query_embedding = self.get_embeddings([query], "query")[0]
        document_embeddings = self.get_embeddings(documents, "reference")
        return query_embedding@document_embeddings.t()

    def select_topk(self, query: str, documents: List[str], k=1) -> torch.Tensor:
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = []
        for i in range((len(documents) + self.max_batch_size - 1) // self.max_batch_size):
            scores.append(self.score_documents_on_query(query, documents[self.max_batch_size*i:self.max_batch_size*(i+1)]).to('cpu'))
        scores = torch.concat(scores)
        return scores.topk(min(k, len(scores))).indices

    def __call__(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        texts = [item['text'] for item in paragraphs]
        topk = self.select_topk(query, texts, topk)
        indices = list(topk.detach().cpu().numpy())
        return [paragraphs[idx] for idx in indices]

