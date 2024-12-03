import typer
from rich.table import Table
from rich import Console

console = Console()
app = typer.Typer()

@app.command()
def start():
    """
    Starts ollama server and ensures the default LLM model is available.
    """
    pass

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
    pass

@app.command()
def read(file_path: str = typer.Argument(..., help="Path of the file to read")):
    """ 
    Reads and starts a chat using the given document from the current working directory.
    """

    pass

@app.command()
def use(model_name: str = typer.Argument(..., help="Name of the model to use from Ollama.")):
    """
    Selects the model to use for conversing with the document.
    """
    
    pass