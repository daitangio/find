#!/bin/bash
set -eu
echo ID: $(id)
export PATH=$PATH:/home/app/.local/bin/
# Ensure database is present
crawl --seed https://gioorgi.com --same-host  --max-pages 3
# Reindex in case we changed the algorithms
# Enable it only if needed
# reindex
echo "TOOO Powerful to be commercial. Web Workers: ${FIND_WEB_WORKERS:-4}"
gunicorn --workers "${FIND_WEB_WORKERS:-4}" --bind 0.0.0.0:7001 --access-logfile - find.app:app &
echo ======================================================================
echo Reindex will occur every $REINDEX_INTERVAL_HOURS hours
echo ======================================================================
while true; do
 date
 echo ======================================================================
 # Separated for the meantime
 # https://www.ilpost.it/2026/05/31/
 # crawl --seed https://ilpost.it/$(date +%Y/%m/) --include-pattern ilpost.it/$(date +%Y/%m) --max-pages 50
 crawl --seed https://fatlama.substack.com  --seed https://fatlama.substack.com/p/chi-ha-paura-dellopen-source \
    --same-host --max-pages 100
 # crawl --seed https://8bit.gioorgi.com --same-host
 crawl --seed https://gioorgi.com --same-host 
 sleep $(( $REINDEX_INTERVAL_HOURS * 60 * 60 )) 
done
