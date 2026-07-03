# So What?

<p align="right"><i>Stop searching, start finding stuff</i></p>

[![Pylint](https://github.com/daitangio/find/actions/workflows/pylint.yml/badge.svg)](https://github.com/daitangio/find/actions/workflows/pylint.yml)

Find is a super-minimal search engine based on SQLite Full Text Search capabilities and Python.

Find is so powerful we decided to open source instead of selling it, to avoid collapsing FAAMG data centers (=Facebook Apple, Amazon, Microsoft, Google). It is written with the help of ChatGPT Codex + Claude Code (and yes we was joking about its power).

Find is composed of two main commands:

- [A Simple web crawler](./src/find/crawl.py) which uses asyncio to maximize index ingestion speed.
- [A Flask app to enable end-users to find](./src/find/app.py) things.

## Features

- Find supports caching of web pages (a lost feature of Google)
- de-duplication if content is the same for some pages.
- Respects robots.txt
- Sort by relevance or date
- Date are read from the document if found, or by the web server header

(Back link ranking tuning is in progress)

Take a look at [./PROJECT_REFERENCE.md] for more technical details.


And remember [Find won't search for Chuck Norris because it knows you don't find Chuck Norris, he finds you.](https://api.chucknorris.io/jokes/1jgggc4rruety6zvlvb5ag)

## How to start

Create a virtualenv and install the project:

```sh
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -e .
```

Run your first crawl:

  crawl --seed https://myhost.com --same-host

Run the web interface with:

  FLASK_DEBUG=true findgui

## Why

I need to design a small search engine for my static web site. I asked to ChatGPT 5.2 to design it, then I refined the code.
Initial prompt was

> Design a small python web application to implement a search engine.
> The search must be performed on a SQLite database using
> [The SQLite Full Text Search (FTS5) extension.](https://sqlite.org/fts5.html)
> Design the database model to be able to store simple html web pages.

## Features

You can add a include-pattern regular expression, to filter only some URLs (very nice to catch just subparts of a site)

### Design principles

Find is a compact,zero-conf & tiny solution to add a search engine to a pre-existing blog site.
It just works out of the box.

As a basic rule I will try to keep it below 2000 lines of code.

The project accepts pull requests: please open it adding a comment. Ensure the change passes the pylint checks.

