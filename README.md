# So What?

Find is a super-minimal search engine based on SQLite Full Text Search capabilities and Python.
It is composed of two modules:

- [A Super simple web crawler](./crawl.py)
- [A Flask app to search](./app.py) the content.

# How to start

Create a virtualenv and install the requirements:

    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt

Run your first crawl:

    ./crawl.py --seed https://myhost.com --concurrency 6 --same-host


# Why

I need to design a small search engine for my static web site. I asked to ChatGPT 5.2 to design it, then I refined the code.
Initial prompt was

    Design a small python web application to implement a search engine. The search must be performed on a SQLite database using the SQLite Full Text Search (FTS5) extension. 
    Design the database model to be able to store simple html web pages.

# How

[SQLite has a full text search capability called FTS5](https://sqlite.org/fts5.html) which offers out of the box also stemming for english language.

ChatGPT for the crawler proposed asyncio I/O (aiohttp & aiosqlite libraries), which is a very good approach to scale the crawler: downloading web pages is a very I/O bound activity and it benefits from a non-blocking library.

Initial implementation has a locking problem: we solved it with a mono-writer database task. 
SQLite is so fast you have an hard time to tune the writer queue: it is very difficult to saturate it.
To avoid data loss, I opted for a queue 4x the concurrency level.

The crawler has a default delay to avoid overloading the target site. For this reason, it is pointless to have too much concurrency if your default delay is high.

The overall project aims to be very compact (*less is more* mantra)

# Next Step and Roadmap

1) The links table is collected but not used on the search right now. The idea is to use it to refine the PageRank. To have an idea try:

    ```sql
    SELECT p.url, COUNT(*) AS out_links
    FROM links l JOIN pages p ON p.id = l.from_page_id
    GROUP BY p.id
    ORDER BY out_links DESC
    LIMIT 20;
    ```
2) Dockerfile+compose is needed to provide easy installation
3) Ability to partial reindex
3) Ability to classify categories and tags on the full text search can be useful for faceting and classification.
"Auto discovery" of the taxonomies can be further idea


Be happy!