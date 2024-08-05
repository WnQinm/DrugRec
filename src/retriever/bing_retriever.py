from .base_retriever import BaseRetriever, SearchResult
from playwright.sync_api import sync_playwright
from typing import List


class BingRetriever(BaseRetriever):
    def __init__(self, model_args, device="cpu", scorer_max_batch_size=400) -> None:
        super().__init__(model_args, device, scorer_max_batch_size)

    def get_search_result(self, question: str) -> List[SearchResult]:
        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()
            try:
                page.goto(f"https://www.bing.com/search?q={question}")
            except:
                page.goto(f"https://www.bing.com")
                page.fill('input[name="q"]', question)
                page.press('input[name="q"]', "Enter")
            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except:
                pass
            # page.wait_for_load_state('networkidle')
            search_results = page.query_selector_all(".b_algo h2")
            for result in search_results:
                title = result.inner_text()
                a_tag = result.query_selector("a")
                if not a_tag:
                    continue
                url = a_tag.get_attribute("href")
                if not url:
                    continue
                # print(title, url)
                results.append(SearchResult(title=title, url=url))
            browser.close()
        return results
