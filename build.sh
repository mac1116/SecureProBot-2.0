#!/bin/bash
set -e

# Install Python 3.11
echo "Installing Python 3.11..."
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Use Python 3.11 for pip
/usr/bin/python3.11 -m pip install --upgrade pip setuptools wheel
/usr/bin/python3.11 -m pip install -r requirements.txt

echo "Build complete!"
