#!/bin/bash
# Download models via Ollama

set -e

echo "📥 Downloading models via Ollama..."
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Starting Ollama service..."
    sudo systemctl start ollama || {
        echo "❌ Cannot start Ollama. Please start manually: sudo systemctl start ollama"
        exit 1
    }
    sleep 3
fi

echo "📥 Pulling Qwen2.5-7B model via Ollama..."
ollama pull qwen2.5:7b

echo ""
echo "✅ Model downloaded via Ollama!"
echo ""
echo "📋 Available models:"
ollama list
echo ""
echo "💡 To use Ollama with RAG service:"
echo "   USE_OLLAMA=true docker compose up"
echo ""
echo "💡 Models location: ~/.ollama/models/"
echo ""

