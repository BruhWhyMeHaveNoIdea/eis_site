from django.db import models

# Create your models here
class Tenders(models.Model):
    icz = models.CharField(
        max_length=100,
        unique=True,
        verbose_name='ИКЗ',
        db_index=True,
        help_text='Идентификационный код закупки'
    )
    object_name = models.CharField(
        max_length=500,
        verbose_name='Наименование объекта закупки',
        db_index=True
    )


    customer_full_name = models.CharField(
        max_length=500,
        verbose_name='Заказчик'
    )
    region = models.CharField(
        max_length=200,
        verbose_name='Регион',
        db_index=True
    )


    procurement_method = models.CharField(
        max_length=200,
        verbose_name='Способ определения поставщика'
    )
    stage = models.CharField(
        max_length=100,
        verbose_name='Этап закупки'
    )
    start_date_time = models.DateTimeField(
        verbose_name='Начало подачи заявок'
    )
    end_date_time = models.DateTimeField(
        verbose_name='Окончание подачи заявок'
    )
    results_date = models.DateField(
        verbose_name='Дата подведения итогов'
    )


    initial_price = models.DecimalField(
        max_digits=15, decimal_places=2,
        verbose_name='НМЦК, ₽'
    )


    contract_execution_period = models.CharField(
        max_length=100,
        verbose_name='Срок исполнения контракта'
    )


    participant_requirements = models.TextField(
        verbose_name='Требования к участникам'
    )
    advantages = models.TextField(
        blank=True, null=True,
        verbose_name='Преимущества (СМП, СОНО и т.д.)'
    )


    delivery_place = models.TextField(
        verbose_name='Место поставки/выполнения работ'
    )


    bid_security_required = models.BooleanField(
        default=False,
        verbose_name='Обеспечение заявки'
    )
    bid_security_amount = models.CharField(
        max_length=100, blank=True, null=True,
        verbose_name='Размер обеспечения заявки'
    )
    performance_required = models.BooleanField(
        default=True,
        verbose_name='Обеспечение исполнения контракта'
    )
    performance_security_size = models.CharField(
        max_length=50, blank=True, null=True,
        verbose_name='Размер обеспечения исполнения'
    )


    email = models.EmailField(verbose_name='E-mail заказчика')
    phone = models.CharField(max_length=50, verbose_name='Телефон')


    electronic_platform_name = models.CharField(
        max_length=200,
        verbose_name='Электронная площадка'
    )
    electronic_platform_url = models.URLField(
        blank=True, null=True,
        verbose_name='Ссылка на площадку'
    )

    class Meta:
        verbose_name = 'Закупка (44-ФЗ)'
        verbose_name_plural = 'Закупки (44-ФЗ)'
        ordering = ['-end_date_time']
        indexes = [
            models.Index(fields=['icz']),
            models.Index(fields=['region']),
            models.Index(fields=['end_date_time']),
        ]

    def __str__(self):
        return f"{self.object_name[:50]} – {self.icz}"