FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    FLASK_DEBUG=0 \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

ENV PATH="$PATH:/home/app/.local/bin"

RUN addgroup --gid 1000  app && adduser --uid 1000 --ingroup app  app
USER app

WORKDIR /home/app

COPY LICENSE .
COPY src src
COPY tests tests
COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir -e .

RUN python3 -m unittest discover -s tests
EXPOSE 5000

CMD ["/home/app/.local/bin/findgui"]
