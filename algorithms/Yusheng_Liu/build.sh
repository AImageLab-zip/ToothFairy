#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t toothfairy_generic:v1.0 "$SCRIPTPATH"