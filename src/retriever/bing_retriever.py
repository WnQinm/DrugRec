from .base_retriever import BaseRetriever, SearchResult
import requests
from bs4 import BeautifulSoup as bs
import time
import random
from typing import List, Optional


class BingRetriever(BaseRetriever):
    def __init__(self, model_args, device="cpu", scorer_max_batch_size=400) -> None:
        super().__init__(model_args, device, scorer_max_batch_size)

    def _kw(self, drug_name, first):
        return {'q':f'what+drug+is+{drug_name}', 'first':first, 'FORM':'PERE1'}

    def get_search_result(self, question: str, min_search_result_num: Optional[int]=None) -> List[SearchResult]:
        if min_search_result_num is None:
            while True:
                try:
                    response = requests.get(
                        "https://cn.bing.com/search",
                        params=self._kw(question, 1),
                        headers=self.headers,
                        timeout=20,
                        verify=False
                    )
                    break
                except:
                    time.sleep(random.randint(1, 8))
            response = bs(response.text, "html.parser")
            results = map(
                lambda result: result.find("a", _ctf="rdr_T"),
                response.find('ol', id="b_results").find_all("div", class_="b_algoheader")
            )
            results = [SearchResult(result.find('h2').get_text(), result.attrs['href']) for result in results]
            return results
        else:
            results = []
            first = 1
            while len(results) < min_search_result_num:
                while True:
                    try:
                        response = requests.get(
                            "https://cn.bing.com/search",
                            params=self._kw(question, 1),
                            headers=self.headers,
                            timeout=20,
                            verify=False
                        )
                        if response.status_code != 200:
                            continue
                        break
                    except:
                        time.sleep(random.randint(1, 8))
                response = bs(response.text, "html.parser")
                temp = map(
                    lambda result: result.find("a", _ctf="rdr_T"),
                    response.find('ol', id="b_results").find_all("div", class_="b_algoheader")
                )
                temp = [SearchResult(result.find('h2').get_text(), result.attrs['href']) for result in temp]
                if len(temp) == 0:
                    break
                else:
                    results.extend(temp)
                first += 10
                time.sleep((random.random()+0.5)*2)
            return results
