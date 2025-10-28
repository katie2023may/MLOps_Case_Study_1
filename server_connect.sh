#!/bin/bash
# This script connects to a remote server via SSH.
# Usage: ./server_connect.sh <username> <server_address>

PORT=2222
KEY=$HOME/.ssh/CS553_keys/group2_key

ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no group2@melnibone.wpi.edu
