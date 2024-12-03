import os
import time
import psutil
import subprocess
import platform
from rich.console import Console
import requests

console = Console()

class OllamaManager:
    
    @staticmethod
    def is_server_running() -> bool:
        """Checks if the Ollama server is running at the given URL."""
        
        url: str = "http://localhost:11434"
        
        try:
            response = requests.get(url, timeout=3)  # Adding a timeout to prevent indefinite hanging
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
                result = subprocess.run(["sudo", "systemctl", "start", "ollama.service"], check=True)
                return result.returncode == 0

            elif os_type == "Darwin":
                # For macOS, run Ollama in the background using subprocess
                process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return process is not None and process.poll() is None

            elif os_type == "Windows":
                process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                return process is not None and process.poll() is None

            else:
                console.print("[red]Unsupported operating system. Ollama service could not be started.[/red]")
                return False

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to start Ollama service: {e}[/red]")
            return False

        except FileNotFoundError:
            console.print("[red]Error: Ollama is not installed or not in PATH[/red]")
            console.print("Please install Ollama following the instructions at: [cyan]https://ollama.ai/download[/cyan]")
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
                subprocess.run(["sudo", "systemctl", "stop", "ollama.service"], check=True, timeout=10)
                return True
            
            elif os_type == "Darwin":
                # For macOS, find and kill the process
                result = subprocess.run(["pkill", "-f", "ollama"], check=True, timeout=10)
                return result.returncode == 0
            
            elif os_type == "Windows":
                try:
                    # Find and terminate all Ollama-related processes
                    for proc in psutil.process_iter(['name']):
                        if proc.info['name'] in ["ollama.exe", "ollama-app.exe"]:
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
                console.print("[red]Unsupported operating system. Cannot stop Ollama service.[/red]")
                return False

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to stop Ollama service: {e}[/red]")
            return False

        except FileNotFoundError:
            console.print("[yellow]No running Ollama server found or command not found![/yellow]")
            return False

        except subprocess.TimeoutExpired:
            console.print("[red]Stopping Ollama service timed out.[/red]")
            return False

        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
            return False
