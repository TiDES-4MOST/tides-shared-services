#!/bin/bash
set -e

# Ensure required directories exist (permissions already correct)
mkdir -p /media/snid_template_options

# Start the service
exec su -s /bin/bash sniduser -c "$*"


