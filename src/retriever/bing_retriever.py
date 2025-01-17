from .base_retriever import BaseRetriever, SearchResult
import requests
from bs4 import BeautifulSoup as bs
import time
import random
from typing import List, Optional, Union


class BingRetriever(BaseRetriever):
    def __init__(self, model_args, **scorer_args) -> None:
        super().__init__(model_args, **scorer_args)

    def _kw(self, question: Union[str, List[str]], first=1):
        if isinstance(question, str):
            return {'q':f'what+drug+is+{question}', 'first':first, 'FORM':'Z9FD1'}
        elif isinstance(question, list):
            return {'q':f'icd+10+{question[0]}+{question[1].replace(" ", "+")}+Symptoms', 'first':first, 'FORM':'Z9FD1'}

    def _sleep(self):
        time.sleep(random.random()*2)

    def _search_bing(self, question: Union[str, List[str]], first:int=1):
        while True:
            self._sleep()
            try:
                response = requests.get(
                    "https://cn.bing.com/search",
                    params=self._kw(question, first),
                    headers=self.headers,
                    timeout=20,
                )
                if response.status_code != 200:
                    continue
                response = bs(response.text, "html.parser")
                results = map(
                    lambda result: result.find("a", _ctf="rdr_T"),
                    response.find('ol', id="b_results").find_all("div", class_="b_algoheader")
                )
                results = [SearchResult(result.find('h2').get_text(), result.attrs['href']) for result in results]
                break
            except:
                continue
        return results

    def get_search_result(
        self,
        question: Union[str, List[str]],
        min_search_result_num: Optional[int] = None,
    ) -> List[SearchResult]:
        if min_search_result_num is None:
            results = self._search_bing(question)
            return results
        else:
            results = []
            first = 1
            while len(results) < min_search_result_num:
                self._sleep()
                temp = self._search_bing(question, first)
                if len(temp) == 0:
                    continue
                else:
                    results.extend(temp)
                first += 10
            return results
