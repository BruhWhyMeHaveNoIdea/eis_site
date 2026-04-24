from scrapling.spiders import Spider, Response
from cfg.selectors import *


class LinkSpider(Spider):
    '''Ассинхронный паук для парсинга ссылок по заданным фильтрам'''

    name = 'link_spider'
    start_urls = []    # Переопределяем при инициализации объекта

    def __init__(self, start_urls,  *args, **kwargs):
        super(LinkSpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls  # Берем из объекта Parser()

    async def parse(self, response: Response):
        '''Парсим ссылки на все тендеры'''

        blocks_tender = response.find_all('div', class_ = BLOCK_TENDER)   # Тут все блоки
        for tender in blocks_tender:
            in_block = tender.find('div', class_= IN_BLOCK)
            link = in_block.find('a::attr(href)').get()    # Тут получаем саму ссылку
            yield {"link": link}

        