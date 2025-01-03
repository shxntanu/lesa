# Copyright 2025 Shantanu Wable
#
# Code Credits: Anish Dabhane (github/@Spartan-71)
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

import os
import time
import json
import psutil
import subprocess
import platform
import requests

from pathlib import Path
from typing import Dict
from platformdirs import user_config_dir, user_cache_dir

from langchain_ollama import OllamaLLM

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class OllamaManager:

    # Use platform-specific config directory
    CONFIG_DIR = Path(user_config_dir("lesa"))
    CACHE_DIR = Path(user_cache_dir("lesa"))

    # Config files
    CONFIG_FILE = CONFIG_DIR / "config.json"
    MODELS_FILE = CONFIG_DIR / "models.json"

    DEFAULT_MODELS = {
        "qwen:4b": {
            "description": "Light(er) model, performant at conversations.",
            "size": "2.3GB",
            "status": "disabled",
            "downloaded": "no",
        },
        "llama3.1:latest": {
            "description": "High-performance model with a larger context window, suitable for larger documents.",
            "size": "4.7GB",
            "status": "disabled",
            "downloaded": "no",
        },
    }

    @classmethod
    def ensure_config(cls) -> None:
        """
        Ensures configuration directory and files exist with default values.
        Creates them if they don't exist.
        """
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize config file if it doesn't exist
        if not cls.CONFIG_FILE.exists():
            cls.save_config({"model_name": "llama3.1:latest"})

        # Initialize models file if it doesn't exist
        if not cls.MODELS_FILE.exists():
            cls.save_models(cls.DEFAULT_MODELS)

    @classmethod
    def get_config(cls) -> Dict:
        """
        Retrieves current configuration.

        Returns:
            Dict: Current configuration settings
        """
        cls.ensure_config()
        # print("Config file location: ", cls.CONFIG_FILE)
        return json.loads(cls.CONFIG_FILE.read_text())

    @classmethod
    def get_models(cls) -> Dict:
        """
        Retrieves available model configurations.

        Returns:
            Dict: Dictionary of available models and their settings
        """
        cls.ensure_config()
        return json.loads(cls.MODELS_FILE.read_text())

    @classmethod
    def save_config(cls, config: Dict) -> None:
        """
        Saves configuration settings to file.

        Args:
            config (Dict): Configuration settings to save
        """
        cls.CONFIG_FILE.write_text(json.dumps(config, indent=2))

    @classmethod
    def save_models(cls, models: Dict) -> None:
        """
        Saves model configurations to file.

        Args:
            models (Dict): Model configurations to save
        """
        cls.MODELS_FILE.write_text(json.dumps(models, indent=2))

    @classmethod
    def get_active_model(cls) -> str:
        """
        Returns the currently active model name.

        Returns:
            str: Name of the active model
        """
        config = cls.get_config()
        return config.get("model_name", "llama3.1:latest")  # Default if not set

    @staticmethod
    def is_server_running() -> bool:
        """Checks if the Ollama server is running at the given URL."""

        url: str = "http://localhost:11434"

        try:
            response = requests.get(
                url, timeout=3
            )  # Adding a timeout to prevent indefinite hanging
            if response.status_code == 200:
                # console.print("[green]Ollama server is running.[/green]")
                return True
            else:
                # console.print(f"[yellow]Ollama server is not running. Status code: {response.status_code}[/yellow]")
                return False

        except requests.ConnectionError:
            # console.print("[red]Ollama server is not reachable. Connection error.[/red]")
            return False

        except requests.Timeout:
            # console.print("[red]Ollama server request timed out.[/red]")
            return False

        except requests.RequestException as e:
            # console.print(f"[red]Unexpected error while checking server: {str(e)}[/red]")
            return False

    @staticmethod
    def start_ollama_service() -> bool:
        """Starts the Ollama service based on the operating system."""

        os_type = platform.system()

        try:
            if os_type == "Linux":
                # For Linux, use systemd to start the service
                result = subprocess.run(
                    ["sudo", "systemctl", "start", "ollama.service"], check=True
                )
                return result.returncode == 0

            elif os_type == "Darwin":
                # For macOS, run Ollama in the background using subprocess
                process = subprocess.Popen(
                    ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                return process is not None and process.poll() is None

            elif os_type == "Windows":
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                )
                return process is not None and process.poll() is None

            else:
                console.print(
                    "[red]Unsupported operating system. Ollama service could not be started.[/red]"
                )
                return False

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to start Ollama service: {e}[/red]")
            return False

        except FileNotFoundError:
            console.print("[red]Error: Ollama is not installed or not in PATH[/red]")
            console.print(
                "Please install Ollama following the instructions at: [cyan]https://ollama.ai/download[/cyan]"
            )
            return False

        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
            return False

    @staticmethod
    def stop_ollama_service() -> bool:
        """Stops the Ollama service based on the current operating system."""
        os_type = platform.system()

        try:
            if os_type == "Linux":
                # For Linux, use systemd to stop the service
                subprocess.run(
                    ["sudo", "systemctl", "stop", "ollama.service"],
                    check=True,
                    timeout=10,
                )
                return True

            elif os_type == "Darwin":
                # For macOS, find and kill the process
                result = subprocess.run(
                    ["pkill", "-f", "ollama"], check=True, timeout=10
                )
                return result.returncode == 0

            elif os_type == "Windows":
                try:
                    # Find and terminate all Ollama-related processes
                    for proc in psutil.process_iter(["name"]):
                        if proc.info["name"] in ["ollama.exe", "ollama-app.exe"]:
                            try:
                                proc.terminate()
                                # Wait a bit for graceful termination
                                proc.wait(timeout=5)
                            except psutil.NoSuchProcess:
                                pass
                            except psutil.TimeoutExpired:
                                # Force kill if not terminated
                                proc.kill()

                    return True

                except Exception as e:
                    print(f"Error stopping Ollama: {e}")
                    return False

            else:
                console.print(
                    "[red]Unsupported operating system. Cannot stop Ollama service.[/red]"
                )
                return False

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to stop Ollama service: {e}[/red]")
            return False

        except FileNotFoundError:
            console.print(
                "[yellow]No running Ollama server found or command not found![/yellow]"
            )
            return False

        except subprocess.TimeoutExpired:
            console.print("[red]Stopping Ollama service timed out.[/red]")
            return False

        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
            return False

    @staticmethod
    def is_model_present(model_name: str) -> bool:
        """
        Checks if a specific model is present in the output of `ollama list`.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is present, False otherwise.

        Raises:
            ValueError: If model_name is empty or not a string.
        """

        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        try:
            # Run the ollama list command with timeout
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False,  # Don't raise CalledProcessError, handle manually
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if result.returncode != 0:
                return False

            models_list = [
                line.strip() for line in result.stdout.split("\n") if line.strip()
            ]

            for model_line in models_list:
                if model_line.split()[0].strip() == model_name:
                    return True

            return False

        except FileNotFoundError:
            console.print(
                f"[yellow]ollama command not found. Please ensure Ollama is installed and in PATH.[/yellow]"
            )
            return False

        except Exception as e:
            console.print(
                f"[red]Unexpected error checking for model '{model_name}': {str(e)}[/red]"
            )
            return False

    @classmethod
    def pull_model(cls, model_name: str, timeout: float = 600.00) -> bool:
        """
        Pulls an Ollama model if it's not already present.

        Args:
            model_name (str): The name of the model to pull
            timeout (Optional[float]): Maximum time in seconds to wait for the pull.
            Defaults to 10 minutes.

        Returns:
            bool: True if model is available (pulled successfully or already present),
                False if pull failed
        """

        try:
            # Check if model is already pulled
            present: bool = OllamaManager.is_model_present(model_name)
            if present:
                console.print(
                    f"[green]Model {model_name} is already pulled and ready to use.[/green]"
                )
                return True

            # Model needs to be pulled
            console.print(f"[cyan]Pulling {model_name}...[/cyan]")
            console.print(
                "NOTE: The download time varies based on your internet speed and the model size.\nIf the download doesn't complete within 10 minutes, please try running the command again."
            )
            time.sleep(1)

            # Create a simple spinner with elapsed time
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {model_name}...", total=None)

                try:
                    # Run the pull command with timeout
                    result = subprocess.run(
                        ["ollama", "pull", model_name],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                        ),
                    )

                    if result.returncode == 0:
                        # Update the table
                        models = cls.get_models()
                        models[model_name]["downloaded"] = "yes"
                        cls.save_models(models)

                        console.print(
                            f"[green]Successfully pulled {model_name}![/green]"
                        )
                        return True
                    else:
                        error_message = (
                            result.stderr if result.stderr else "Unknown error"
                        )
                        console.print(
                            f"[red]Error pulling model: {error_message.strip()}[/red]"
                        )
                        return False

                except subprocess.TimeoutExpired as e:
                    # Clean up the process when timeout occurs
                    if hasattr(e, "process"):
                        e.process.kill()
                    console.print(
                        f"[red]Error: Pull operation timed out after {timeout} seconds[/red]"
                    )
                    return False

                except KeyboardInterrupt:
                    console.print("\n[yellow]Download cancelled![/yellow]")
                    return False

        except FileNotFoundError:
            console.print(
                "[red]Error: ollama command not found. Please ensure Ollama is installed and in PATH[/red]"
            )
            return False
        except Exception as e:
            console.print(f"[red]Unexpected error while pulling model: {str(e)}[/red]")
            return False

    @classmethod
    def delete_model(cls, model_name: str) -> bool:
        """
        Deletes an Ollama model if it's already present.

        Args:
            model_name (str): The name of the model to delete

        Returns:
            bool: True if model deleted successfully, False if model doesn't exist
        """
        try:
            # Delete the model
            # console.print(f"[yellow]Deleting {model_name}...[/yellow]")
            result = subprocess.run(
                ["ollama", "rm", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if result.returncode == 0:
                # Update the models table
                models = cls.get_models()
                models[model_name]["downloaded"] = "no"
                cls.save_models(models)
                console.print(f"[green]Successfully deleted {model_name}.[/green]")
                return True
            else:
                error = result.stderr.strip()
                console.print(f"[red]Error deleting {model_name}: {error}[/red]")
                return False

        except FileNotFoundError:
            console.print(
                "[red]Error: ollama command not found. Please ensure Ollama is installed and in PATH.[/red]"
            )
            return False
        except Exception as e:
            console.print(f"[red]Unexpected error while deleting model: {str(e)}[/red]")
            return False

    @classmethod
    def serve_llm(cls) -> OllamaLLM:
        """
        Serve an Ollama LLM model for inference.

        Args:
            model_name (str): The name of the model to serve

        Returns:
            OllamaLLM: An instance of OllamaLLM for inference
        """
        return OllamaLLM(
            model=cls.get_active_model(),
            temperature=0.9,
        )
