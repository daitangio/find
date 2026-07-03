# Python development

Install on Install your virtualenv with

```sh
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e .
```

If you prefer uv just run

```sh
    uv sync
```

## Extra skills for codex

```sh
    rtk init --codex
    npx skills add JuliusBrussee/caveman -a codex
```

# Unit test Run

See etc/pre-commit script for sanity checks you can use before pushing your changes

    
# Dockerized test run

To just populate a throw-away database:

FIND_HOME=/tmp docker compose up --build

## AI Tips

To check token usage use something like

    npx ccusage 

## Caveman skills

To install caveman we used

    npx skills add JuliusBrussee/caveman -a codex
    # Also install rtk command to further reduce token usage
    brew install rtk    