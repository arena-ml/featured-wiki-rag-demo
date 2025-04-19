#!/bin/bash

(ollama serve > /proc/1/fd/1 2>/proc/1/fd/2 &)

sleep 15

# Pass control to the command the user specified (if any)
exec "$@"
