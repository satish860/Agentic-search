"""Contract reader tool with line number support, similar to Claude's Read tool."""

import os
import glob
import chardet
from pathlib import Path
from typing import Optional, List, Dict, Union


class ContractReader:
    """Read contracts with line number support, similar to Claude's Read tool."""
    
    def __init__(self, contracts_dir: str = "contracts"):
        """Initialize the contract reader.
        
        Args:
            contracts_dir: Directory containing contract files (default: "contracts")
        """
        self.contracts_dir = Path(contracts_dir)
        self._file_cache = {}  # Cache for file info to improve performance
    
    def read(self, file_path: str, offset: int = 1, limit: Optional[int] = None) -> str:
        """Read a contract file with line numbers, similar to Claude's Read tool.
        
        Args:
            file_path: Path to the contract file (can be relative to contracts_dir or absolute)
            offset: Starting line number (1-based, default=1)
            limit: Number of lines to read (default=None for all remaining lines)
            
        Returns:
            String with line numbers in format: "line_number→content"
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If offset is less than 1 or limit is less than 1
        """
        if offset < 1:
            raise ValueError("Offset must be 1 or greater")
        
        if limit is not None and limit < 1:
            raise ValueError("Limit must be 1 or greater")
        
        # Resolve file path
        full_path = self._resolve_file_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Contract file not found: {file_path}")
        
        # Detect encoding
        encoding = self._detect_encoding(full_path)
        
        # Read lines with offset and limit
        lines = []
        current_line = 1
        
        try:
            with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
                for line in f:
                    if current_line >= offset:
                        # Format line number similar to Claude's Read tool
                        formatted_line = f"{current_line}->{line.rstrip()}"
                        lines.append(formatted_line)
                        
                        # Check if we've read enough lines
                        if limit is not None and len(lines) >= limit:
                            break
                    
                    current_line += 1
            
            return '\n'.join(lines)
            
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}") from e
    
    def read_section(self, file_path: str, start: int, end: int) -> str:
        """Read from start to end line (inclusive).
        
        Args:
            file_path: Path to the contract file
            start: Starting line number (1-based, inclusive)
            end: Ending line number (1-based, inclusive)
            
        Returns:
            String with line numbers in format: "line_number→content"
        """
        if start < 1 or end < 1:
            raise ValueError("Start and end line numbers must be 1 or greater")
        
        if start > end:
            raise ValueError("Start line must be less than or equal to end line")
        
        limit = end - start + 1
        return self.read(file_path, offset=start, limit=limit)
    
    def get_file_info(self, file_path: str) -> Dict[str, Union[int, str, float]]:
        """Get file metadata: total lines, size, encoding.
        
        Args:
            file_path: Path to the contract file
            
        Returns:
            Dictionary with file information
        """
        full_path = self._resolve_file_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Contract file not found: {file_path}")
        
        # Check cache first
        cache_key = str(full_path)
        file_stat = full_path.stat()
        
        if (cache_key in self._file_cache and 
            self._file_cache[cache_key]['mtime'] == file_stat.st_mtime):
            return self._file_cache[cache_key]['info']
        
        # Calculate file info
        encoding = self._detect_encoding(full_path)
        total_lines = 0
        
        try:
            with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = 0
        
        file_info = {
            'total_lines': total_lines,
            'size_bytes': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
            'encoding': encoding,
            'name': full_path.name,
            'path': str(full_path)
        }
        
        # Cache the result
        self._file_cache[cache_key] = {
            'info': file_info,
            'mtime': file_stat.st_mtime
        }
        
        return file_info
    
    def list_contracts(self, pattern: str = "*.txt") -> List[str]:
        """List all contract files matching pattern.
        
        Args:
            pattern: Glob pattern to match files (default: "*.txt")
            
        Returns:
            List of contract file names
        """
        if not self.contracts_dir.exists():
            return []
        
        # Use glob to find matching files
        search_pattern = str(self.contracts_dir / pattern)
        matching_files = glob.glob(search_pattern)
        
        # Return just the file names, sorted
        file_names = [Path(f).name for f in matching_files]
        return sorted(file_names)
    
    def search_in_file(self, file_path: str, pattern: str, context: int = 2) -> List[Dict[str, Union[int, str]]]:
        """Search for pattern in file and return matches with context lines.
        
        Args:
            file_path: Path to the contract file
            pattern: Text pattern to search for (case-insensitive)
            context: Number of context lines before and after each match
            
        Returns:
            List of dictionaries with match information
        """
        full_path = self._resolve_file_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Contract file not found: {file_path}")
        
        encoding = self._detect_encoding(full_path)
        matches = []
        
        try:
            with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            
            pattern_lower = pattern.lower()
            
            for i, line in enumerate(lines, 1):
                if pattern_lower in line.lower():
                    # Calculate context range
                    start_ctx = max(1, i - context)
                    end_ctx = min(len(lines), i + context)
                    
                    # Extract context lines
                    context_lines = []
                    for j in range(start_ctx - 1, end_ctx):  # -1 because enumerate starts at 1
                        line_num = j + 1
                        formatted_line = f"{line_num}->{lines[j].rstrip()}"
                        context_lines.append(formatted_line)
                    
                    matches.append({
                        'match_line': i,
                        'match_text': line.strip(),
                        'context_start': start_ctx,
                        'context_end': end_ctx,
                        'context': '\n'.join(context_lines)
                    })
            
            return matches
            
        except Exception as e:
            raise Exception(f"Error searching in file {file_path}: {str(e)}") from e
    
    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve file path, checking both absolute and relative to contracts_dir.
        
        Args:
            file_path: File path to resolve
            
        Returns:
            Resolved Path object
        """
        path = Path(file_path)
        
        # If it's already absolute and exists, return it
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to contracts directory
        contracts_path = self.contracts_dir / file_path
        if contracts_path.exists():
            return contracts_path
        
        # If neither exists, return the contracts_dir version for error handling
        return contracts_path
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding string
        """
        try:
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'
            
            # Common fallbacks
            if encoding.lower() in ['ascii', 'windows-1252']:
                encoding = 'utf-8'
            
            return encoding
            
        except Exception:
            # Fallback to utf-8 if detection fails
            return 'utf-8'