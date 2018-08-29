#!/usr/bin/env bash

npm install
nohup npm start >out.file 2>&1 &
