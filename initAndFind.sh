#!/bin/bash
echo ID: $(id)
export PATH=$PATH:/home/app/.local/bin/
findgui &
while true; do
 sleep 5
 date
 # Separated for the meantime
 crawl --seed https://8bit.gioorgi.com --same-host
 crawl --seed https://gioorgi.com --same-host 
 sleep $(( 24 * 60 * 60 ))
done
