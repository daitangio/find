FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=find.app:app \
    FLASK_ENV=production \
    FLASK_RUN_HOST=0.0.0.0 \
    FIND_WEB_WORKERS=4
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

# RUN pip install pylint
# RUN pylint $(git ls-files '*.py')

COPY initAndFind.sh .
CMD ["./initAndFind.sh"]
