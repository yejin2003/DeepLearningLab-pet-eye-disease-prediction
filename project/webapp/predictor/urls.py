from __future__ import annotations

from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict/", views.predict_api, name="predict_api"),
]
