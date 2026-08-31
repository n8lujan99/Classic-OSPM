#!/usr/bin/env bash

INTERVAL="${1:-10}"
echo "[WATCH] Looking for active OSPM/Python process owned by $USER"
while true; do
    PID="$(ps -u "$USER" -o pid=,pcpu=,args= |
        grep -E 'python.*OSPM|python.*daemon|python.*start|julia' |
        grep -v -E 'grep|watch_ospm' |
        sort -k2 -nr |
        awk 'NR==1 {print $1}')"
    if [[ -z "$PID" ]]; then
        echo
        echo "[WATCH] $(date '+%F %T') no active OSPM process found"
        sleep "$INTERVAL"
        continue
    fi
    echo
    echo "================================================================================================"
    echo "[WATCH] $(date '+%F %T') PID=$PID"
    ps -p "$PID" -o pid,ppid,etime,%cpu,%mem,nlwp,rss,vsz,stat,wchan:32,cmd
    echo
    echo "[THREADS — highest CPU]"
    ps -L -p "$PID" -o tid,psr,pcpu,stat,wchan:28,comm --sort=-pcpu | head -12
    if [[ -r "/proc/$PID/io" ]]; then
        echo
        echo "[IO]"
        grep -E 'rchar|wchar|read_bytes|write_bytes' "/proc/$PID/io"
    fi
    sleep "$INTERVAL"
done