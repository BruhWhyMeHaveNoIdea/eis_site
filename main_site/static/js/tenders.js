/**
 * Скрипт для страницы тендеров (44-ФЗ)
 */

// Данные тендеров для модального окна (заполняются в шаблоне)
window.tendersData = window.tendersData || {};

// Кэш ML-результатов
window.mlResultsCache = window.mlResultsCache || {};

/**
 * Открыть модальное окно с деталями тендера
 * @param {number} tenderId - ID тендера
 */
async function openModal(tenderId) {
    const modal = document.getElementById('tenderModal');
    const content = document.getElementById('modalContent');
    const data = window.tendersData[tenderId];
    
    if (!data) {
        content.innerHTML = '<p class="text-red-500">Данные не найдены</p>';
        modal.classList.remove('hidden');
        return;
    }
    
    // Основная информация
    const mainInfo = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">ИКЗ</p>
                <p class="font-mono text-sm">${escapeHtml(data.icz)}</p>
            </div>
            
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">Наименование объекта закупки</p>
                <p class="font-semibold">${escapeHtml(data.object_name)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Заказчик</p>
                <p>${escapeHtml(data.customer_full_name)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Регион</p>
                <p>${escapeHtml(data.region)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Способ определения</p>
                <p>${escapeHtml(data.procurement_method)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Этап закупки</p>
                <p><span class="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">${escapeHtml(data.stage)}</span></p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Начало подачи заявок</p>
                <p>${escapeHtml(data.start_date_time)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Окончание подачи заявок</p>
                <p class="font-semibold text-red-600">${escapeHtml(data.end_date_time)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Дата подведения итогов</p>
                <p>${escapeHtml(data.results_date)}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">НМЦК</p>
                <p class="text-lg font-bold text-green-600">${escapeHtml(data.initial_price)} ₽</p>
            </div>
            
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">Срок исполнения контракта</p>
                <p>${escapeHtml(data.contract_execution_period)}</p>
            </div>
            
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">Место поставки</p>
                <p>${escapeHtml(data.delivery_place)}</p>
            </div>
            
            ${data.advantages ? `
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">Преимущества</p>
                <p class="text-green-700">${escapeHtml(data.advantages)}</p>
            </div>
            ` : ''}
            
            <div>
                <p class="text-sm text-gray-500">Обеспечение заявки</p>
                <p>${data.bid_security_required ? 'Да' : 'Нет'} ${data.bid_security_amount ? `(${escapeHtml(data.bid_security_amount)})` : ''}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Обеспечение исполнения</p>
                <p>${data.performance_required ? 'Да' : 'Нет'} ${data.performance_security_size ? `(${escapeHtml(data.performance_security_size)})` : ''}</p>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Email</p>
                <a href="mailto:${escapeHtml(data.email)}" class="text-blue-600 hover:underline">${escapeHtml(data.email)}</a>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Телефон</p>
                <a href="tel:${escapeHtml(data.phone)}" class="text-blue-600 hover:underline">${escapeHtml(data.phone)}</a>
            </div>
            
            <div>
                <p class="text-sm text-gray-500">Электронная площадка</p>
                <p>${escapeHtml(data.electronic_platform_name)}</p>
            </div>
            
            <div class="col-span-1 md:col-span-2">
                <p class="text-sm text-gray-500">Требования к участникам</p>
                <p class="text-sm text-gray-600">${escapeHtml(data.participant_requirements)}</p>
            </div>
        </div>
    `;
    
    // Загрузка ML-результатов
    let mlContent = '';
    if (window.mlResultsCache[tenderId]) {
        mlContent = renderMLResults(window.mlResultsCache[tenderId]);
    } else {
        mlContent = `
            <div class="border-t pt-4">
                <div id="mlLoading" class="text-center py-8">
                    <i class="fa-solid fa-spinner fa-spin text-3xl text-purple-600"></i>
                    <p class="mt-2 text-gray-600">Выполняется анализ тендера...</p>
                </div>
            </div>
        `;
    }
    
    content.innerHTML = mainInfo + mlContent;
    
    // Обновляем заголовок
    const modalTitle = document.getElementById('modalTitle');
    if (modalTitle) {
        modalTitle.textContent = data.object_name.substring(0, 80) + (data.object_name.length > 80 ? '...' : '');
    }
    
    modal.classList.remove('hidden');
    
    // Загружаем ML-результаты если нет в кэше
    if (!window.mlResultsCache[tenderId]) {
        loadMLResults(tenderId);
    }
}

/**
 * Загрузка ML-результатов через API
 * @param {number} tenderId - ID тендера
 */
async function loadMLResults(tenderId) {
    const mlLoading = document.getElementById('mlLoading');
    
    try {
        const response = await fetch(`/api/analyze/${tenderId}/`);
        const data = await response.json();
        console.log('data,', data)
        
        window.mlResultsCache[tenderId] = data;
        
        if (mlLoading) {
            mlLoading.outerHTML = renderMLResults(data);
        }
    } catch (error) {
        if (mlLoading) {
            mlLoading.outerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                    <i class="fa-solid fa-circle-exclamation text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-700">Ошибка ML-анализа: ${escapeHtml(error.message)}</p>
                </div>
            `;
        }
    }
}

/**
 * Рендеринг ML-результатов
 * @param {Object} data - Данные ML-анализа
 * @returns {string} HTML
 */
function renderMLResults(data) {
    if (data.error) {
        return `
            <div class="border-t pt-4">
                <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p class="text-red-700">${escapeHtml(data.error)}</p>
                </div>
            </div>
        `;
    }
    
    let html = '<div class="border-t pt-4">';
    html += '<h4 class="text-lg font-bold mb-3 text-purple-700"><i class="fa-solid fa-robot mr-2"></i>ML-анализ</h4>';
    
    // Статус анализатора
    if (data.analyzer_trained !== undefined) {
        html += `
            <div class="mb-4">
                <span class="px-2 py-1 text-xs rounded-full ${data.analyzer_trained ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}">
                    <i class="fa-solid ${data.analyzer_trained ? 'fa-check-circle' : 'fa-exclamation-circle'} mr-1"></i>
                    ${data.analyzer_trained ? 'Модель обучена' : 'Модель не обучена'}
                </span>
            </div>
        `;
    }
    
    // Информация о тендере
    if (data.tender_info) {
        const ti = data.tender_info;
        html += `
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                <h5 class="font-bold text-gray-800 mb-2">
                    <i class="fa-solid fa-file-contract mr-2"></i>Анализируемый тендер
                </h5>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                    <div class="md:col-span-2">
                        <span class="text-gray-500">Наименование:</span>
                        <p class="font-medium">${escapeHtml(ti.name || '')}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Заказчик:</span>
                        <p>${escapeHtml(ti.customer || '')}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Регион:</span>
                        <p>${escapeHtml(ti.region || '')}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">НМЦК:</span>
                        <p class="font-bold text-green-600">${ti.price ? `${ti.price} ₽` : 'Н/Д'}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Способ определения:</span>
                        <p>${escapeHtml(ti.procurement_method || '')}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Прогнозирование цены
    if (data.price_prediction && !data.price_prediction.error) {
        const pp = data.price_prediction;
        html += `
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
                <h5 class="font-bold text-purple-800 mb-2">
                    <i class="fa-solid fa-chart-line mr-2"></i>Прогноз цены
                </h5>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                    <div>
                        <span class="text-gray-500">Прогноз:</span>
                        <p class="font-bold text-green-600">${pp.predicted_price ? `${Number(pp.predicted_price).toLocaleString('ru-RU')} ₽` : 'Н/Д'}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Метод:</span>
                        <p class="font-medium">${escapeHtml(pp.method || 'Н/Д')}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Доверие:</span>
                        <p class="font-medium">${pp.confidence ? `${(pp.confidence * 100).toFixed(1)}%` : 'Н/Д'}</p>
                    </div>
                    ${pp.price_range && pp.price_range.length === 2 ? `
                    <div>
                        <span class="text-gray-500">Диапазон:</span>
                        <p class="font-medium">${Number(pp.price_range[0]).toLocaleString('ru-RU')} - ${Number(pp.price_range[1]).toLocaleString('ru-RU')} ₽</p>
                    </div>
                    ` : ''}
                    <div>
                        <span class="text-gray-500">Похожих найдено:</span>
                        <p class="font-medium">${pp.similar_tenders_used} / ${pp.similar_tenders_total}</p>
                    </div>
                </div>
                
                ${pp.similar_tenders_preview && pp.similar_tenders_preview.length > 0 ? `
                <div class="mt-3">
                    <p class="text-xs text-gray-500 mb-1">Похожие тендеры для прогноза:</p>
                    <ul class="text-xs space-y-1">
                        ${pp.similar_tenders_preview.map(t => `
                            <li class="bg-white rounded px-2 py-1">
                                <span class="text-green-600 font-medium">${t.price ? `${Number(t.price).toLocaleString('ru-RU')} ₽` : ''}</span>
                                <span class="text-gray-400 mx-1">•</span>
                                <span class="text-blue-600">${t.similarity ? `( ${(t.similarity * 100).toFixed(0)}%)` : ''}</span>
                                <br>
                                <span class="text-gray-600 truncate">${escapeHtml(t.name || '')}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    // Похожие тендеры

    if (data.similar_tenders && data.similar_tenders.length > 0) {
        html += `
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <h5 class="font-bold text-blue-800 mb-2">
                    <i class="fa-solid fa-magnifying-glass mr-2"></i>Похожие тендеры
                </h5>
                <div class="space-y-2">
                    ${data.similar_tenders.map((t, i) => `
                        <div class="bg-white rounded-lg p-3 text-sm">
                            <div class="flex justify-between items-start mb-1">
                                <span class="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs font-medium">
                                    #${t.rank || i + 1} • ${(t.similarity_score * 100).toFixed(0)}%
                                </span>
                                ${t.cluster && t.cluster !== 'N/A' ? `<span class="text-gray-400 text-xs">Кластер: ${escapeHtml(t.cluster)}</span>` : ''}
                            </div>
                            <p class="font-medium text-gray-900 mb-1 truncate" title="${escapeHtml(t.tender_name || '')}">${escapeHtml(t.tender_name || '')}</p>
                            <div class="grid grid-cols-2 gap-2 text-xs text-gray-600">
                                <span><i class="fa-solid fa-building mr-1"></i>${t.customer && t.customer !== 'N/A' ? escapeHtml(t.customer) : 'Н/Д'}</span>
                                <span><i class="fa-solid fa-location-dot mr-1"></i>${t.region && t.region !== 'N/A' ? escapeHtml(t.region) : 'Н/Д'}</span>
                                <span><i class="fa-solid fa-ruble-sign mr-1"></i>${formatPrice(t.price) || 'Н/Д'}</span>
                                <span><i class="fa-solid fa-file-contract mr-1"></i>${t.procurement_method && t.procurement_method !== 'N/A' ? escapeHtml(t.procurement_method) : 'Н/Д'}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Метаданные поиска
    if (data.search_metadata) {
        const sm = data.search_metadata;
        html += `
            <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-4">
                <h5 class="font-bold text-gray-700 mb-2 text-sm">
                    <i class="fa-solid fa-circle-info mr-1"></i>Метаданные поиска
                </h5>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                    <div>
                        <span class="text-gray-500">Алгоритм:</span>
                        <p class="font-medium">${escapeHtml(sm.algorithm || '')}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Найдено:</span>
                        <p class="font-medium">${sm.results_returned || 0}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Всего проверено:</span>
                        <p class="font-medium">${sm.total_tenders_searched || 0}</p>
                    </div>
                    <div>
                        <span class="text-gray-500">Время:</span>
                        <p class="font-medium">${data.metadata?.execution_time_seconds ? `${data.metadata.execution_time_seconds.toFixed(2)}с` : 'Н/Д'}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Общие метаданные
    if (data.metadata) {
        const m = data.metadata;
        html += `
            <div class="mt-4 text-xs text-gray-400 text-center">
                <span>Исторических тендеров: ${m.historical_tenders_count || 0}</span>
                ${m.execution_time_seconds ? `<span class="mx-2">•</span><span>Время анализа: ${m.execution_time_seconds.toFixed(2)}с</span>` : ''}
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

/**
 * Закрыть модальное окно
 */
function closeModal() {
    document.getElementById('tenderModal').classList.add('hidden');
}

/**
 * Экранирование HTML-символов для безопасности
 * @param {string} text - Текст для экранирования
 * @returns {string} - Экранированный текст
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

/**
 * Инициализация кнопки парсера
 */
function initParserButton() {
    const parseBtn = document.getElementById('parseBtn');
    if (!parseBtn) return;
    
    parseBtn.addEventListener('click', async function() {
        const btn = this;
        btn.disabled = true;
        btn.textContent = 'Обновление...';
        
        try {
            const response = await fetch('/api/start-parcer');
            
            // Проверяем Content-Type
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error("Сервер вернул не JSON (ошибка " + response.status + ")");
            }
            
            const data = await response.json();
            
            if (response.ok) {
                const msg = `Данные обновлены!\n+ Создано: ${data.created}\n~ Обновлено: ${data.updated}` + 
                           (data.errors_count > 0 ? `\n! Ошибок: ${data.errors_count}` : '');
                alert(msg);
                location.reload();
            } else if (response.status === 409) {
                alert('Парсер уже запущен. Пожалуйста, дождитесь завершения.');
            } else {
                alert('Ошибка: ' + (data.error || 'Неизвестная ошибка'));
            }
        } catch (e) {
            alert('Ошибка соединения: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Обновить данные';
        }
    });
}

/**
 * Форматирование цены из русского формата (15 766,67) в число
 * @param {string|number} price - Цена
 * @returns {string|null} - Отформатированная цена или null
 */
function formatPrice(price) {
    if (!price || price === 'N/A' || price === 'Н/Д') {
        return null;
    }
    
    // Преобразуем в строку
    let priceStr = String(price).trim();
    
    // Заменяем запятую на точку и удаляем пробелы
    priceStr = priceStr.replace(/\s/g, '').replace(',', '.');
    
    const num = parseFloat(priceStr);
    
    if (isNaN(num)) {
        return null;
    }
    
    return num.toLocaleString('ru-RU') + ' ₽';
}

/**
 * Инициализация модального окна
 */
function initModal() {
    const modal = document.getElementById('tenderModal');
    if (!modal) return;
    
    // Закрытие по клику вне модального окна
    modal.addEventListener('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });
    
    // Закрытие по Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

/**
 * Инициализация при загрузке страницы
 */
document.addEventListener('DOMContentLoaded', function() {
    initParserButton();
    initModal();
});
