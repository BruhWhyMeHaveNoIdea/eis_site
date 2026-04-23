from scrapling.spiders import Spider, Response
from other.text_cleaner import simple_clean
from cfg.selectors import *
from cfg.get_links import LinksCollector


class TenderSpider(Spider):
    '''Основной асинхронный паук. Парсим содержимое тендеров из всех полученных ссылок'''

    name = "tender_spider"
    start_urls = []    # Переопределяем при инициализации объекта

    def __init__(self, start_page: int = 1, end_page: int = 1, per_page: int = 10, pub_date: str = "", close_date: str = "", *args, **kwargs):
        super(TenderSpider, self).__init__(*args, **kwargs)
        collector = LinksCollector(
            start_page=start_page,
            end_page=end_page,
            per_page=per_page,
            pub_date=pub_date,
            close_date=close_date
        )
        self.start_urls = collector.collect_tenders_links()  # Берём из art/links.json

    async def parse(self, response: Response):
        '''Основная функция парсинга, Т.к. великолепные верстальщики не оставили отдельных
        селекторов под каждый пункт, то парсим все секции, а потом перебираем каждую'''

        blocks = response.find_all('section')    # Вот тут получаем все секции
        data = {}

        i = 1
        for block in blocks:
            if block.find('span', class_= SECTION_TITLE):
                title = simple_clean(block.find('span', class_= SECTION_TITLE).text)    # Вот тут получаем название секции
            else:
                title = f'unnamed{i}'
                i += 1    # На случай если название секции отстутствует

            if block.find('span', class_= SECTION_INFO):
                info = simple_clean(block.find('span', class_= SECTION_INFO).text)    # Вот тут получаем смысловую нагрузку
                data[title] = info

        yield {**data}

