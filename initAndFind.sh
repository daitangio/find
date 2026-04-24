#!/bin/bash
set -eu
echo ID: $(id)
export PATH=$PATH:/home/app/.local/bin/
findgui &
echo Reinde will occur every $REINDEX_INTERVAL_HOURS hours
while true; do
 sleep 5
 date
 # Separated for the meantime
 crawl --seed https://8bit.gioorgi.com --same-host
 crawl --seed https://gioorgi.com --same-host 
 sleep $(( $REINDEX_INTERVAL_HOURS * 60 * 60 ))
done
