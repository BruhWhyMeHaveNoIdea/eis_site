from django.urls import path, include
import page.views as views

urlpatterns = [
    path('', views.index),
    path('api/start-parcer', views.start_parser),
    path('api/analyze/<int:tender_id>/', views.analyze_tender_api),
    path('api/ml-cache/clear/', views.clear_ml_cache, name='clear_ml_cache_all'),
    path('api/ml-cache/clear/<int:tender_id>/', views.clear_ml_cache, name='clear_ml_cache_tender'),
]