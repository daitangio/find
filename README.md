# So What?

Find is a super-minimal search engine based on SQLite Full Text Search capability and Python.
It is composed of two modules:

- A Super simple web crawler
- A Flask app

# How to start

create a virtualenv and install the requirements:

    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt

Run your fist crawl:

    ./crawl.py --seed https://myhost.com --concurrency 6 --same-host


# Why

I need to design a small search engine for my static web site. I asked to ChatGPT 5.2 to design it, then I refined the code.
Initial prompt was

    Design a small python web application to implement a search engine. The search must be performed on a SQLite database using the SQLite Full Text Search (FTS5) extension. 
    Design the database model to be able to store simple html web pages.


Be happy!