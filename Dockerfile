FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (pillow 등 이미지 처리용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY project/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY project /app/project

WORKDIR /app/project/webapp

ENV PYTHONUNBUFFERED=1
ENV DEBUG=False
ENV ALLOWED_HOSTS=*

# Hugging Face Spaces는 7860 포트를 기본으로 사용
EXPOSE 7860

CMD ["sh", "-c", "python manage.py collectstatic --noinput && gunicorn pet_eye_site.wsgi --bind 0.0.0.0:7860 --workers 1 --timeout 180"]
