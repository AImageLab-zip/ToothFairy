#!/bin/bash
set -e

# Check if running as root, if so, fix permissions and switch to appuser
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, fixing permissions and switching to appuser..."
    
    # Ensure directories exist and are writable by appuser
    mkdir -p /input /output
    chown appuser:appgroup /input /output
    chmod 755 /input /output
    
    # Execute the command as appuser
    exec su appuser -c "python -m evaluation"
else
    # Already running as non-root user
    echo "Running as user: $(id)"
    exec python -m evaluation
fi
