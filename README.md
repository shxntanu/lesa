![Banner](assets/banner.png)

> _lesa_ (verb) - to read, to study, to learn

**_lesa_** is a CLI tool built in Python that allows you to converse with your documents from the terminal, completely offline and on-device using **Ollama**. Open the terminal in the directory of your choice and start a conversation with any document!

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
![PyPI - Version](https://img.shields.io/pypi/v/lesa)
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/lesa) -->

</div>

> [!NOTE]
> This project is unfinished and in active development. Pull requests, suggestions and any sort of help in general is always welcome :)

## 🚀 Features

-   🖥️ **Completely On-Device**: Uses Ollama under the hood to interface with LLMs, so you can be sure your data is not leaving your device.
-   📚 **Converse with (almost) all documents**: Supports PDF, DOCX and Text files.
-   🤖 **Wide Range of LLMs**: Choose the Large Language Model of your choice. Whether you want to keep it quick and concise, or want to go all in with a huge context window, the choice is yours.

## 🔧 Setup

### Prerequisites

This project uses [Ollama](https://ollama.com/) under the hood to utilize the power of large language models. To install Ollama, run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This project uses the Llama 3.1 8b or Qwen 4b model as the default. You can use any other model as well, just make sure it has enough context window to understand the content of your documents.

Pull Llama or Qwen using:

```bash
ollama pull qwen:4b
```

or

```bash
ollama pull llama3.1
```

### Installation

Simply install the package using pip:

```bash
pip install lesa
```

To upgrade to the latest version, run:

```bash
pip install -U lesa
```
