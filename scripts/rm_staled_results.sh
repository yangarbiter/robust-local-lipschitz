#!/usr/bin/env bash

find $1 -name "*.pkl" -type 'f' -size -40c # -delete
