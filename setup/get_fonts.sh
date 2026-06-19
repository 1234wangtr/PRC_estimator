#!/bin/bash
set -e
# install Times New Roman font and clear old matplotlib cache
sudo apt update
sudo apt install ttf-mscorefonts-installer
fc-cache -f -v
rm -rf ~/.cache/matplotlib