#!/usr/bin/env bash

function check_python_version_gte_3_6 {
    echo "Checking for >= python3.6"
    hash python3.6  2>/dev/null \
    || hash python3.7 2>/dev/null;
}

function do_install_for_linux {
    echo "Installing ZDoom dependencies"
    sudo apt install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip cmake

    # vizdoom dependencies
    sudo apt install libboost-all-dev

    echo ""
    echo "-- dependencies to support external environments have been installed --"
    echo ""
}

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "Detected linux"
    do_install_for_linux
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi