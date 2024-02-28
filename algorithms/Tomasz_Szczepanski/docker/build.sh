#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t  toothfairy_algorithm "$SCRIPTPATH"
