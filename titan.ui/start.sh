#!/usr/bin/env bash

npm install
nohup npm >out.file 2>&1 &
