window.tendersData = window.tendersData || {};

/**
 * Открыть модальное окно с деталями тендера
 * @param {number} tenderId - ID тендера
 */
function openModal(tenderId) {
    const modal = document.getElementById('tenderModal');
    const content = document.getElementById('modalContent');
    const data = window.tendersData[tenderId];
    
    if (!data) {
        content.innerHTML = '<p class="text-red-500">Данные не найдены</p>';
        modal.classList.remove('hidden');
        return;
    }
    
    content.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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
    
    modal.classList.remove('hidden');
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
 * Примечание: На странице тендеров используется tenders.js с улучшенной версией
 */
function initParserButton() {
    const parseBtn = document.getElementById('parseBtn');
    if (!parseBtn) return;
    
    // Устанавливаем флаг, чтобы tenders.js не вешал дублирующий обработчик
    parseBtn.dataset.parserInitialized = 'true';
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