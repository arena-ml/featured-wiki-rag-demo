#!/bin/bash

# Start Ollama in background if not running
pgrep -f "ollama serve" > /dev/null || nohup ollama serve > /var/log/ollama.log 2>&1 &

# Launch shell
exec /bin/bash
