#!/usr/bin/env bash

function check_python_version_gte_3_6 {
    echo "Checking for >= python3.6"
    hash python3.6  2>/dev/null \
    || hash python3.7 2>/dev/null;
}

function do_install_for_linux {
    echo "Installing python dependencies"
    sudo apt-get install graphviz cmake clang
    pip install --upgrade pip==21.0.1
    pip install -e .
    pip install pydot
    AutoROM

    echo "Installing Mujoco dependencies"
    sudo add-apt-repository ppa:jamesh/snap-support
    sudo apt-get update && sudo apt-get install libosmesa6-dev patchelf -y
    mkdir $HOME/.mujoco
    wget -P $HOME/.mujoco https://www.roboti.us/file/mjkey.txt
    wget -P /tmp https://roboti.us/download/mujoco200_linux.zip
    unzip /tmp/mujoco200_linux.zip -d $HOME/.mujoco/ && mv $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin" >> $HOME/.bashrc
    source $HOME/.bashrc
    rm /tmp/mujoco200_linux.zip

    echo "Install StarCraftII environments"
    sh malib/envs/star_craft2/install.sh

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