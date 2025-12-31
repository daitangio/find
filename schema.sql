PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Latest version per URL
CREATE TABLE IF NOT EXISTS pages (
  id            INTEGER PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,        -- normalized canonical URL
  title         TEXT,
  html          TEXT NOT NULL,
  text          TEXT NOT NULL,
  content_hash  TEXT NOT NULL,               -- sha256 of html
  status_code   INTEGER,
  fetched_at    TEXT                         -- ISO timestamp
);

-- All historical versions (only added when content_hash changes)
CREATE TABLE IF NOT EXISTS page_versions (
  id            INTEGER PRIMARY KEY,
  page_id       INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  content_hash  TEXT NOT NULL,
  title         TEXT,
  html          TEXT NOT NULL,
  text          TEXT NOT NULL,
  status_code   INTEGER,
  fetched_at    TEXT,                        -- ISO timestamp
  UNIQUE(page_id, content_hash)
);

-- Link graph edges: from page -> to_url, plus optional to_page_id when discovered/known
CREATE TABLE IF NOT EXISTS links (
  id            INTEGER PRIMARY KEY,
  from_page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  to_url        TEXT NOT NULL,               -- normalized
  to_page_id    INTEGER REFERENCES pages(id) ON DELETE SET NULL,
  first_seen_at TEXT,
  last_seen_at  TEXT,
  UNIQUE(from_page_id, to_url)
);

CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_page_id);
CREATE INDEX IF NOT EXISTS idx_links_to_url ON links(to_url);
CREATE INDEX IF NOT EXISTS idx_versions_page ON page_versions(page_id);

-- FTS index over latest title/text/url in pages (external content)
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
USING fts5(
  title,
  text,
  url,
  content='pages',
  content_rowid='id',
  tokenize='unicode61'
);

-- Keep FTS in sync with pages
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
  INSERT INTO pages_fts(rowid, title, text, url)
  VALUES (new.id, new.title, new.text, new.url);
END;

CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, text, url)
  VALUES ('delete', old.id, old.title, old.text, old.url);
END;

CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, text, url)
  VALUES ('delete', old.id, old.title, old.text, old.url);
  INSERT INTO pages_fts(rowid, title, text, url)
  VALUES (new.id, new.title, new.text, new.url);
END;
