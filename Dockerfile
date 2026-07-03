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

COPY pyproject.toml .

# GG Codex suggestion to extract requirements to cache them before real install 
# You can comment this 2 lines if you want a more "standard" - unoptimized procedure
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print('\n'.join(p['project']['dependencies']))" > requirements.txt
RUN pip install --user -r requirements.txt
# RUN ls /home/app/.cache/

RUN pip install --upgrade pip

COPY tests tests
COPY src src
COPY README.md .

RUN pip install -e .

RUN python3 -m unittest discover -s tests

# RUN pip install pylint
# RUN pylint $(git ls-files '*.py')

COPY initAndFind.sh .
CMD ["./initAndFind.sh"]
