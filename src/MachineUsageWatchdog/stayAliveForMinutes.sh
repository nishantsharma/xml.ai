#!/bin/bash 
MINUTES_LEFT=$1
until [  $MINUTES_LEFT -le 0 ]; do
    echo Reserved minutes left are $MINUTES_LEFT
    let MINUTES_LEFT=MINUTES_LEFT-1 
    sleep 60
    touch /var/run/MachineUsageWatchdog  
done
