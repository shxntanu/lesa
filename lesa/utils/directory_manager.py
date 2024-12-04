import os
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

class DirectoryManager:
    """
    Configuration manager for Lesa to track file states and changes.
    """
    
    CONFIG_DIR = '.lesa'
    CONFIG_FILE = 'config.json'
    
    def __init__(self, base_path: str = '.'):
        """
        Initialize the configuration manager for a specific directory.
        
        :param base_path: Base directory to manage (default is current directory)
        """
        self.base_path = os.path.abspath(base_path)
        self.config_path = os.path.join(self.base_path, self.CONFIG_DIR)
        self.config_file_path = os.path.join(self.config_path, self.CONFIG_FILE)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        :param file_path: Path to the file
        :return: File hash as a string
        """
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Dict:
        """
        Get metadata for a specific file.
        
        :param file_path: Path to the file
        :return: Dictionary with file metadata
        """
        stat = os.stat(file_path)
        return {
            'path': os.path.relpath(file_path, self.base_path),
            'hash': self._calculate_file_hash(file_path),
            'size': stat.st_size,
            'modified_time': stat.st_mtime
        }
    
    def init(self) -> None:
        """
        Initialize the Lesa configuration for the current directory.
        Creates config directory and initial configuration snapshot.
        """
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_path, exist_ok=True)
        
        file_metadata = self._scan_directory()
        
        config_data = {
            'initialized_at': datetime.now().isoformat(),
            'base_path': self.base_path,
            'files': file_metadata
        }
        
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Lesa embeddings initialized in {self.base_path}")
    
    def _scan_directory(self, ignore_patterns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Scan the directory and collect metadata for all files.
        
        :param ignore_patterns: List of patterns to ignore (e.g., ['.lesa', '.git'])
        :return: Dictionary of file metadata
        """
        ignore_patterns = ignore_patterns or [self.CONFIG_DIR, '.git']
        file_metadata = {}
        
        for root, dirs, files in os.walk(self.base_path):
            dirs[:] = [d for d in dirs if not any(ig in d for ig in ignore_patterns)]
            
            for file in files:
                full_path = os.path.join(root, file)
                
                if not any(ig in full_path for ig in ignore_patterns):
                    try:
                        metadata = self._get_file_metadata(full_path)
                        file_metadata[metadata['path']] = metadata
                    except Exception as e:
                        print(f"Could not process file {full_path}: {e}")
        
        return file_metadata
    
    def check_for_changes(self) -> Dict:
        """
        Check if any files have changed since last initialization.
        
        :return: Dictionary of change types
        """
        # Ensure config file exists
        if not os.path.exists(self.config_file_path):
            raise FileNotFoundError("Lesa embeddings not initialized. Run 'lesa init' first.")
        
        # Read existing configuration
        with open(self.config_file_path, 'r') as f:
            existing_config = json.load(f)
        
        current_files = self._scan_directory()
        
        changes = {
            'new_files': [],
            'deleted_files': [],
            'modified_files': []
        }
        
        # Check for new and modified files
        for path, metadata in current_files.items():
            if path not in existing_config['files']:
                changes['new_files'].append(path)
            elif (existing_config['files'][path]['hash'] != metadata['hash'] or 
                  existing_config['files'][path]['size'] != metadata['size']):
                changes['modified_files'].append(path)
        
        # Check for deleted files
        for path in existing_config['files']:
            if path not in current_files:
                changes['deleted_files'].append(path)
        
        return changes
    
    def update_configuration(self) -> None:
        """
        Update the configuration after detecting changes.
        Rescan the directory and update the configuration file.
        """
        file_metadata = self._scan_directory()
        
        config_data = {
            'initialized_at': datetime.now().isoformat(),
            'base_path': self.base_path,
            'files': file_metadata
        }
        
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print("Configuration updated successfully")