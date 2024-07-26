import asyncio
from playwright.async_api import async_playwright, Page


class FetchRawPage:
    results ={}
    context = None

    @classmethod
    async def one_page_handle(cls, context, url):
        # 开启事件监听
        # page.on('response',printdata)
        # 进入子页面
        try:
            cls.results[url] = None
            response = await context.request.get(url, timeout=5000)
            # 等待子页面加载完毕
            cls.results[url] = await response.text()
        except Exception as e:
            pass

    @classmethod
    async def get_conetent(cls):
        if not cls.context:
            # print("加载驱动")
            playwright = await async_playwright().start()
            browser = await playwright.firefox.launch()
            # 新建上下文
            cls.context = await browser.new_context()
        return cls.context

    @staticmethod
    async def close_browser(browser):
        # 关闭浏览器驱动
        await browser.close()

    @classmethod
    async def get_raw_pages_(cls, context, urls):
        # 封装异步任务
        tasks = []
        cls.results = {}
        for url in urls:
            tasks.append(asyncio.create_task(cls.one_page_handle(context, url)))
        await asyncio.wait(tasks, timeout=10)

    @classmethod
    async def get_raw_pages(cls, urls, close_browser=False):
        context = await cls.get_conetent()
        await cls.get_raw_pages_(context,urls)

