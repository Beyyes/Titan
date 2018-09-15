#!/usr/bin/env bash

wget https://nodejs.org/dist/v6.9.2/node-v6.9.2-linux-x64.tar.xz
xz -d node-v6.9.2-linux-x64.tar.xz
tar -xvf node-v6.9.2-linux-x64.tar

sudo ln node-v6.9.2-linux-x64/bin/node /usr/local/bin
sudo node-v6.9.2-linux-x64/bin/npm install
sudo nohup node-v6.9.2-linux-x64/bin/npm start >out.file 2>&1 &
