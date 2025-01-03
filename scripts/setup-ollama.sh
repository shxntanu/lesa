#!/bin/bash

# Copyright 2025 Shantanu Wable
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
get_os() {
    case "$(uname -s)" in
    Linux*) echo "linux" ;;
    Darwin*) echo "macos" ;;
    *) echo "unsupported" ;;
    esac
}

# Check if Ollama is already installed
if command_exists ollama; then
    echo "Ollama is already installed."
else
    echo "Ollama not found. Installing..."

    OS=$(get_os)
    case $OS in
    "linux")
        curl -fsSL https://ollama.com/install.sh | sh
        ;;
    "macos")
        brew install ollama
        ;;
    *)
        echo "Error: Unsupported operating system"
        exit 1
        ;;
    esac

    # Check if installation was successful
    if ! command_exists ollama; then
        echo "Error: Ollama installation failed"
        exit 1
    fi
    echo "Ollama installed successfully!"
fi

# Start Ollama service if it's not running
if ! pgrep -x "ollama" >/dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5 # Give some time for the service to start
fi

# Pull the llama3.1:latest model
echo "Pulling llama3.1:latest model..."
if ollama pull llama3.1:latest; then
    echo "Model llama3.1:latest pulled successfully!"
else
    echo "Error: Failed to pull llama3.1:latest model"
    exit 1
fi

echo "Setup completed successfully!"
