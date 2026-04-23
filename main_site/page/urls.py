from django.urls import path, include
import page.views as views

urlpatterns = [
    path('', views.index),
    path('parser/start', views.start_parser),
    path('api/analyze/<int:tender_id>/', views.analyze_tender_api),
]