# Python development

Install on Install your virtualenv with

```sh
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e .
```

# Unit test Run

    python3 -m unittest discover -s tests && pylint $(git ls-files '*.py')
    
# Dockerized test run

To just populate a throw-away database:

FIND_HOME=/tmp docker compose up --build