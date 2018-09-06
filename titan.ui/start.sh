#!/usr/bin/env bash

sudo apt-get install npm -y
npm install
nohup npm start >out.file 2>&1 &
