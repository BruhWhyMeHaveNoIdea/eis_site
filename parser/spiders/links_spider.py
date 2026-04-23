from scrapling.spiders import Spider, Response
from cfg.get_links import LinksCollector
from cfg.selectors import *


class LinkSpider(Spider):
    '''Ассинхронный паук для парсинга ссылок по заданным фильтрам
    С учетом пагинации собираем все интересующие нас ссылки.
    
    Параметры передаются при создании объекта:
    - start_page: начальная страница
    - end_page: конечная страница
    - per_page: записей на странице
    - pub_date: дата публикации (DD.MM.YYYY)
    - close_date: дата закрытия (DD.MM.YYYY, опционально)
    '''

    name = 'link_spider'
    start_urls = []    # Переопределяем при инициализации объекта

    def __init__(self, start_page: int, end_page: int, per_page: int, pub_date: str, close_date: str = "", *args, **kwargs):
        super(LinkSpider, self).__init__(*args, **kwargs)
        collector = LinksCollector(
            start_page=start_page,
            end_page=end_page,
            per_page=per_page,
            pub_date=pub_date,
            close_date=close_date
        )
        self.start_urls = collector.collect_pagination()  # Динамически генерируем

    async def parse(self, response: Response):
        '''Парсим ссылки на все тендеры из фильтра. Ввиду своей неопытности и еще чего-то (уже не помню)
        мы парсим "лесенкой" вместо точечного селектора'''

        blocks_tender = response.find_all('div', class_ = BLOCK_TENDER)   # Тут все блоки
        for tender in blocks_tender:
            in_block = tender.find('div', class_= IN_BLOCK)
            link = in_block.find('a::attr(href)').get()    # Тут получаем саму ссылку
            yield {"link": link}

        