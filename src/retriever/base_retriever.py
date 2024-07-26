from .playwright_based_crawl_new import FetchRawPage
from ..model.contriver import ContrieverScorer
import re
from html2text import html2text
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json
import asyncio


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
    def __init__(self, tokenizer_path, retriever_ckpt_path, device=None, scorer_max_batch_size=400) -> None:
        self.loop = asyncio.get_event_loop()
        self.scorer = ContrieverScorer(tokenizer_path, retriever_ckpt_path, device, scorer_max_batch_size)

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

    def _fetch(self, urls: List[str]) -> Dict[str, str]:
        self.loop.run_until_complete(FetchRawPage.get_raw_pages(urls, close_browser=True))
        return {url:text for url,text in FetchRawPage.results.items() if text is not None}

    def _limit_length(self, paragraphs, low, high):
        ret = []
        for item in paragraphs:
            item = item.strip()
            item = re.sub(r"\[\d+\]", "", item)
            item = re.sub(r'\[([^\]]*)\]\(.*?\)', r'\1', item)
            if len(item) < low:
                continue
            if len(item) > high:
                item = item[:high] + "..."
            ret.append(item)
        return ret

    def query(self, question:str, result_length_low:int=50, result_length_high:int=1200, topk:int=5):
        search_results = self._search(question)
        if len(search_results) == 0:
            return None
        url2title = self._pre_handle_search_result(search_results)

        fetch_results = self._fetch(url2title.keys())
        if len(fetch_results) == 0:
            return None

        data_list = []
        for url in fetch_results.keys():
            extract_results = self._limit_length(html2text(fetch_results[url]).split("\n"), result_length_low, result_length_high)
            for value in extract_results:
                data_list.append({
                    "url": url,
                    "title": url2title[url],
                    "text": value
                })
        if len(data_list) == 0:
            return None

        return self.scorer(question, data_list, topk)