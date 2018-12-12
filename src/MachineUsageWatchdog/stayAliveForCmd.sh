#!/bin/sh

# Enable job control.
set -m

# Launch the job to watch.
cmd=$@
$cmd &
cmd_pid=$!
echo "CMD_PID=$cmd_pid"

# Keep the machine alive while the process is running in the background.
while [ -d "/proc/$cmd_pid" ]; do touch /var/run/MachineUsageWatchdog; sleep 2; done &
watch_pid=$!
echo "WATCH_PID=$watch_pid"

# Wait for command to complete.
fg 1
