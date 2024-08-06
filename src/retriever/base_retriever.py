from ..model.bgem3 import M3ForScore
from ..utils.arguments import ModelArguments
import re
import requests
from bs4 import BeautifulSoup as bs
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json


class SearchResult:
    def __init__(self, title, url) -> None:
        self.title = title
        self.url = url

    def dump(self):
        return {
            "title": self.title,
            "url": self.url
        }

    def __str__(self) -> str:
        return json.dumps(self.dump())


class BaseRetriever(ABC):
    def __init__(self, model_args:ModelArguments, device="cpu", scorer_max_batch_size=400) -> None:
        if model_args.model_path:
            self.scorer = M3ForScore(model_args, device=device, batch_size=scorer_max_batch_size)
        self.headers = {
            "content": "text/html; charset=UTF-8",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "cache-control": "no-cache",
            "x-client-data": "eyIxIjoiMCIsIjEwIjoiXCJTNUx6bE5nNUR4UEVtK1FZS0REZ0FJUStqdklQbmpENWVmZFQ2d05ZZVFRPVwiIiwiMiI6IjAiLCIzIjoiMCIsIjQiOiIxNTU2Nzc1MjQyNzgzNjM1NTYyIiwiNSI6IlwiYzZ6dThuYWJiOFFDa0hneFhEenNCZWZpSTVKWEY4MUs0U0tjcXpzK2tjaz1cIiIsIjYiOiJzdGFibGUiLCI3IjoiODU4OTkzNDU5MjAxIiwiOSI6ImRlc2t0b3AifQ",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36 Edg/127.0.0.0",
            'Connection': 'close'
        }

    @abstractmethod
    def get_search_result(self, question: str) -> List[SearchResult]:
        pass

    def _search(self, question:str, max_tries:int=3) -> List[SearchResult]:
        cnt = 0
        while cnt < max_tries:
            cnt += 1
            results = self.get_search_result(question)
            if len(results) > 0:
                return results
        print('No Result')
        return []

    def _pre_handle_search_result(self, results: List[SearchResult]) -> defaultdict[str, str]:
        url2title = defaultdict(str)
        for result in results:
            url = re.sub(r'#.*$', '', result.url)
            if url in url2title.keys() or "http://%s"%url in url2title.keys() or "https://%s"%url in url2title.keys():
                continue
            if not url.startswith("http"):
                url = "http://%s" % url
            url2title[url] = result.title
        return url2title

    def _fetch(self, urls: List[str], passage_len_low=50) -> defaultdict[str, List[str]]:
        fetch_result = defaultdict(list)
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=60, verify=False)
            except:
                continue
            if 'html' not in response.headers['Content-Type']:
                continue
            response = bs(response.text, "html.parser")
            for p in response.find_all("p"):
                p:str = p.get_text()
                p = p.replace("\n", ' ').strip()
                p = re.sub(r'\s+', ' ', p)
                if len(p) > passage_len_low:
                    fetch_result[url].append(p)
        return fetch_result

    def query(self, question:str, result_length_low:int=50):
        search_results = self._search(question)
        if len(search_results) == 0:
            return None
        url2title = self._pre_handle_search_result(search_results)

        fetch_results = self._fetch(url2title.keys(), result_length_low)
        if len(fetch_results) == 0:
            return None

        data_list = []
        for url in fetch_results.keys():
            data_list.extend([{"url":url, "title":url2title[url], "text":text} for text in fetch_results[url]])
        if len(data_list) == 0:
            return None

        return data_list

    def __call__(self, question:str, result_length_low:int=50, topk:int=5):
        data_list = self.query(question, result_length_low)
        return self.scorer(question, data_list, topk)