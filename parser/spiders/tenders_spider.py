from scrapling.spiders import Spider, Response
from other.text_cleaner import simple_clean
from cfg.selectors import *



class TenderSpider(Spider):
    '''Основной асинхронный паук. Парсим содержимое тендеров из всех полученных ссылок'''

    name = "tender_spider"
    start_urls = []    # Переопределяем при инициализации объекта

    def __init__(self, start_urls, *args, **kwargs):
        super(TenderSpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls # Берем из объекта Parser()

    async def parse(self, response: Response):
        '''Основная функция парсинга'''

        blocks = response.find_all('section')    # Вот тут получаем все секции
        data = {}

        for block in blocks:
            i = 1
            if block.find('span', class_= SECTION_TITLE):
                title = simple_clean(block.find('span', class_= SECTION_TITLE).text)    # Вот тут получаем название секции
            else:
                title = f'unnamed{i}'
                i += 1    # На случай если название секции отстутствует

            if block.find('span', class_= SECTION_INFO):
                info = simple_clean(block.find('span', class_= SECTION_INFO).text)    # Вот тут получаем смысловую нагрузку
                data[title] = info

        yield {**data}

