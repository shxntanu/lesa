import typer
import time
import shutil
from datetime import datetime
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from lesa.core.ollama import OllamaManager
from lesa.core.conversations import ConversationManager
from lesa.core.directory_manager import DirectoryManager

cm = ConversationManager()
dm = DirectoryManager()

console = Console()
app = typer.Typer()

@app.command()
def start():
    """
    Starts ollama server and ensures the default LLM model is available.
  
    
    Returns:
        Optional[subprocess.Popen]: Process object if server starts successfully, None otherwise
    """

    console.print(
        Panel(
            Text("📚 Turn your terminal into a File Interpreter", style="bold green", justify="center"),
            border_style="green",
            title="Lesa",
            width=100
        )
    )

    if shutil.which("ollama") is None:
        console.print("[red]Error: Ollama is not installed or not in PATH[/red]")
        console.print("Please install Ollama following the instructions at: [cyan]https://ollama.ai/download[/cyan]")
        raise typer.Exit(1)
 
    # First check if server is already running
    if not OllamaManager.is_server_running():
    
        started : bool = OllamaManager.start_ollama_service()
        
        if started:
            time.sleep(3)
            console.print("[green]Ollama server started successfully![/green]")
        else:
            return None

    else:
        console.print("[yellow]Warning: Ollama server is already running![/yellow]")

    # Check and pull default model
    model_name = "llama3.1:latest"

    # console.print(f"[blue]Checking for default model {model_name}...[/blue]")
    if not OllamaManager.is_model_present(model_name):
        console.print(f"[cyan]Default model {model_name} not found.[/cyan]")

        time.sleep(1)
        if not OllamaManager.pull_model(model_name):
            console.print(f"[red]Failed to pull default model. Please check your internet connection[/red]")
            return None
    else:
        console.print(f"[green]Default model {model_name} is ready![/green]")
    return None

@app.command()
def stop():
    """
    Stops the Ollama Server.
    """
    
    pass

@app.command()
def embed():
    """
    Looks for embeddable files in the current working directory and creates vector embeddings of the same.
    """
    
    console.print(
        Panel(
            Text("📚 Turn your terminal into a File Interpreter", style="bold green", justify="center"),
            border_style="green",
            title="Lesa",
            width=100
        )
    )
    
    if not OllamaManager.is_server_running():
        console.print(f"[red]Error: Ollama server is not running![/red]")
        console.print(f"Start the ollama server by running [cyan]lesa start[/cyan] command.")
        raise typer.Exit(1)

    if not dm.check_configuration_folder():
        console.print("[yellow]Configuration folder not found. Initializing...[/yellow]")
        dm.init()
    
    if not dm.retrieve_config_key("embeddings_initialized"):
        with console.status("Extracting text from files...", spinner="earth") as status:
            for file in dm.files:
                console.log(f"Extracting text from {file}...")
                docs = dm.extract_file_text(filepath=file)
                console.log(f"Embedding documents from {file}...")
                dm.embed_documents(docs)
            console.log("Saving vector embeddings...")
            dm.save_vector_store()
            dm.update_config_key("embeddings_initialized", True)
            dm.update_config_key('embeddings_init_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        console.print("[green]Initialized configuration for embedding files[/green]")
        
    else:
        changes = dm.check_for_changes()
        if len(changes.get('new_files')) == 0 and len(changes.get('modified_files')) == 0 and len(changes.get('deleted_files')) == 0:
            console.print("[yellow]No changes detected in the directory[/yellow]")
        else:
            dm.scan_files()
            console.print("[green]Files in the directory have been changed since last embedding. Embedding again...[/green]")
            with console.status("Extracting text from files...", spinner="earth") as status:
                for file in dm.files:
                    console.log(f"Extracting text from {file}...")
                    docs = dm.extract_file_text(filepath=file)
                    console.log(f"Embedding documents from {file}...")
                    dm.embed_documents(docs)
                console.log("Saving vector embeddings...")
                dm.save_vector_store()
                dm.update_config_key('embeddings_init_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            console.print("[green]Embeddings updated successfully![/green]")

@app.command()
def read(file_path: str = typer.Argument(..., help="Path of the file to read")):
    """ 
    Reads and starts a chat using the given document from the current working directory.
    """
    
    if not OllamaManager.is_server_running():
        console.print(f"[red]Error: Ollama server is not running![/red]")
        console.print(f"Start the ollama server by running [cyan]lesa start[/cyan] command.")
        raise typer.Exit(1)
    
    return cm.embed_single_document_and_chat(file_path)

@app.command()
def chat():
    """
    Starts a chat with the embedded documents.
    """
    
    if not OllamaManager.is_server_running():
        console.print(f"[red]Error: Ollama server is not running![/red]")
        console.print(f"Start the ollama server by running [cyan]lesa start[/cyan] command.")
        raise typer.Exit(1)
    
    return cm.start_conversation()

@app.command()
def use(model_name: str = typer.Argument(..., help="Name of the model to use from Ollama.")):
    """
    Selects the model to use for conversing with the document.
    """
    
    pass

if __name__ == "__main__":
    app()