"""Tests for ContractReader."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.contract_reader import ContractReader


@pytest.fixture
def temp_contracts_dir():
    """Create a temporary directory with test contract files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        test_file1 = temp_path / "test_contract1.txt"
        test_file2 = temp_path / "test_contract2.txt"
        empty_file = temp_path / "empty_contract.txt"
        
        # Test file 1 content
        with open(test_file1, 'w', encoding='utf-8') as f:
            f.write("Line 1: Contract Header\n")
            f.write("Line 2: Party A Information\n")
            f.write("Line 3: Party B Information\n")
            f.write("Line 4: Terms and Conditions\n")
            f.write("Line 5: Payment Terms\n")
            f.write("Line 6: Liability Clause\n")
            f.write("Line 7: Termination Clause\n")
            f.write("Line 8: Governing Law\n")
            f.write("Line 9: Signatures\n")
            f.write("Line 10: End of Contract\n")
        
        # Test file 2 with special characters
        with open(test_file2, 'w', encoding='utf-8') as f:
            f.write("Contract with special chars: €, ©, ™\n")
            f.write("Unicode test: α, β, γ\n")
            f.write("Empty line follows:\n")
            f.write("\n")
            f.write("Line after empty\n")
        
        # Empty file
        empty_file.touch()
        
        yield temp_path


@pytest.fixture
def contract_reader(temp_contracts_dir):
    """Create a ContractReader instance with temporary directory."""
    return ContractReader(str(temp_contracts_dir))


class TestContractReader:
    """Test cases for ContractReader class."""
    
    def test_init_default_directory(self):
        """Test ContractReader initialization with default directory."""
        reader = ContractReader()
        assert reader.contracts_dir == Path("contracts")
    
    def test_init_custom_directory(self, temp_contracts_dir):
        """Test ContractReader initialization with custom directory."""
        reader = ContractReader(str(temp_contracts_dir))
        assert reader.contracts_dir == temp_contracts_dir
    
    def test_read_entire_file(self, contract_reader):
        """Test reading entire file without offset or limit."""
        result = contract_reader.read("test_contract1.txt")
        
        lines = result.split('\n')
        assert len(lines) == 10
        assert lines[0] == "1->Line 1: Contract Header"
        assert lines[4] == "5->Line 5: Payment Terms"
        assert lines[9] == "10->Line 10: End of Contract"
    
    def test_read_with_offset(self, contract_reader):
        """Test reading file with offset."""
        result = contract_reader.read("test_contract1.txt", offset=3)
        
        lines = result.split('\n')
        assert len(lines) == 8  # Lines 3-10
        assert lines[0] == "3->Line 3: Party B Information"
        assert lines[-1] == "10->Line 10: End of Contract"
    
    def test_read_with_limit(self, contract_reader):
        """Test reading file with limit."""
        result = contract_reader.read("test_contract1.txt", limit=3)
        
        lines = result.split('\n')
        assert len(lines) == 3
        assert lines[0] == "1->Line 1: Contract Header"
        assert lines[2] == "3->Line 3: Party B Information"
    
    def test_read_with_offset_and_limit(self, contract_reader):
        """Test reading file with both offset and limit."""
        result = contract_reader.read("test_contract1.txt", offset=5, limit=3)
        
        lines = result.split('\n')
        assert len(lines) == 3
        assert lines[0] == "5->Line 5: Payment Terms"
        assert lines[1] == "6->Line 6: Liability Clause"
        assert lines[2] == "7->Line 7: Termination Clause"
    
    def test_read_special_characters(self, contract_reader):
        """Test reading file with special characters."""
        result = contract_reader.read("test_contract2.txt")
        
        lines = result.split('\n')
        assert "Contract with special chars: €, ©, ™" in lines[0]
        assert "Unicode test: α, β, γ" in lines[1]
    
    def test_read_empty_file(self, contract_reader):
        """Test reading empty file."""
        result = contract_reader.read("empty_contract.txt")
        assert result == ""
    
    def test_read_nonexistent_file(self, contract_reader):
        """Test reading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            contract_reader.read("nonexistent.txt")
    
    def test_read_invalid_offset(self, contract_reader):
        """Test reading with invalid offset raises ValueError."""
        with pytest.raises(ValueError, match="Offset must be 1 or greater"):
            contract_reader.read("test_contract1.txt", offset=0)
        
        with pytest.raises(ValueError, match="Offset must be 1 or greater"):
            contract_reader.read("test_contract1.txt", offset=-1)
    
    def test_read_invalid_limit(self, contract_reader):
        """Test reading with invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be 1 or greater"):
            contract_reader.read("test_contract1.txt", limit=0)
        
        with pytest.raises(ValueError, match="Limit must be 1 or greater"):
            contract_reader.read("test_contract1.txt", limit=-1)
    
    def test_read_offset_beyond_file(self, contract_reader):
        """Test reading with offset beyond file length."""
        result = contract_reader.read("test_contract1.txt", offset=20)
        assert result == ""
    
    def test_read_section(self, contract_reader):
        """Test read_section method."""
        result = contract_reader.read_section("test_contract1.txt", 3, 5)
        
        lines = result.split('\n')
        assert len(lines) == 3
        assert lines[0] == "3->Line 3: Party B Information"
        assert lines[1] == "4->Line 4: Terms and Conditions"
        assert lines[2] == "5->Line 5: Payment Terms"
    
    def test_read_section_invalid_range(self, contract_reader):
        """Test read_section with invalid range."""
        with pytest.raises(ValueError, match="Start and end line numbers must be 1 or greater"):
            contract_reader.read_section("test_contract1.txt", 0, 5)
        
        with pytest.raises(ValueError, match="Start line must be less than or equal to end line"):
            contract_reader.read_section("test_contract1.txt", 5, 3)
    
    def test_get_file_info(self, contract_reader):
        """Test get_file_info method."""
        info = contract_reader.get_file_info("test_contract1.txt")
        
        assert info['total_lines'] == 10
        assert info['size_bytes'] > 0
        assert info['size_mb'] >= 0
        assert info['encoding'] == 'utf-8'
        assert info['name'] == 'test_contract1.txt'
        assert 'path' in info
    
    def test_get_file_info_nonexistent(self, contract_reader):
        """Test get_file_info with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            contract_reader.get_file_info("nonexistent.txt")
    
    def test_list_contracts(self, contract_reader):
        """Test list_contracts method."""
        contracts = contract_reader.list_contracts()
        
        assert len(contracts) == 3
        assert "test_contract1.txt" in contracts
        assert "test_contract2.txt" in contracts
        assert "empty_contract.txt" in contracts
    
    def test_list_contracts_with_pattern(self, contract_reader):
        """Test list_contracts with custom pattern."""
        contracts = contract_reader.list_contracts("*contract1*")
        
        assert len(contracts) == 1
        assert contracts[0] == "test_contract1.txt"
    
    def test_list_contracts_empty_directory(self):
        """Test list_contracts with nonexistent directory."""
        reader = ContractReader("nonexistent_dir")
        contracts = reader.list_contracts()
        
        assert contracts == []
    
    def test_search_in_file(self, contract_reader):
        """Test search_in_file method."""
        matches = contract_reader.search_in_file("test_contract1.txt", "Payment", context=1)
        
        assert len(matches) == 1
        match = matches[0]
        assert match['match_line'] == 5
        assert "Payment Terms" in match['match_text']
        assert match['context_start'] == 4
        assert match['context_end'] == 6
        
        # Check context includes surrounding lines
        context_lines = match['context'].split('\n')
        assert len(context_lines) == 3
        assert "4->Line 4: Terms and Conditions" in context_lines
        assert "5->Line 5: Payment Terms" in context_lines
        assert "6->Line 6: Liability Clause" in context_lines
    
    def test_search_in_file_case_insensitive(self, contract_reader):
        """Test search_in_file is case insensitive."""
        matches = contract_reader.search_in_file("test_contract1.txt", "PAYMENT")
        
        assert len(matches) == 1
        assert "Payment Terms" in matches[0]['match_text']
    
    def test_search_in_file_no_matches(self, contract_reader):
        """Test search_in_file with no matches."""
        matches = contract_reader.search_in_file("test_contract1.txt", "nonexistent")
        
        assert matches == []
    
    def test_search_in_file_nonexistent(self, contract_reader):
        """Test search_in_file with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            contract_reader.search_in_file("nonexistent.txt", "test")
    
    def test_file_caching(self, contract_reader):
        """Test that file info is cached properly."""
        # First call should populate cache
        info1 = contract_reader.get_file_info("test_contract1.txt")
        
        # Second call should use cache
        info2 = contract_reader.get_file_info("test_contract1.txt")
        
        assert info1 == info2
        assert len(contract_reader._file_cache) == 1
    
    def test_resolve_file_path_absolute(self, contract_reader, temp_contracts_dir):
        """Test _resolve_file_path with absolute path."""
        abs_path = temp_contracts_dir / "test_contract1.txt"
        resolved = contract_reader._resolve_file_path(str(abs_path))
        
        assert resolved == abs_path
        assert resolved.exists()
    
    def test_resolve_file_path_relative(self, contract_reader, temp_contracts_dir):
        """Test _resolve_file_path with relative path."""
        resolved = contract_reader._resolve_file_path("test_contract1.txt")
        expected = temp_contracts_dir / "test_contract1.txt"
        
        assert resolved == expected
        assert resolved.exists()
    
    def test_detect_encoding_utf8(self, contract_reader):
        """Test encoding detection for UTF-8 file."""
        encoding = contract_reader._detect_encoding(
            contract_reader.contracts_dir / "test_contract2.txt"
        )
        assert encoding == 'utf-8'
    
    @patch('builtins.open', side_effect=OSError("Mocked file error"))
    def test_read_file_error_handling(self, mock_open, contract_reader):
        """Test error handling when file reading fails."""
        with pytest.raises(Exception, match="Error reading file"):
            contract_reader.read("test_contract1.txt")


class TestContractReaderIntegration:
    """Integration tests with actual contract files."""
    
    def test_read_actual_contract_file(self):
        """Test reading actual contract files from contracts directory."""
        # This test only runs if contracts directory exists
        contracts_dir = Path("contracts")
        if not contracts_dir.exists():
            pytest.skip("No contracts directory found")
        
        reader = ContractReader()
        contracts = reader.list_contracts()
        
        if contracts:
            # Test with first available contract
            contract_name = contracts[0]
            
            # Test basic reading
            result = reader.read(contract_name, limit=5)
            lines = result.split('\n')
            assert len(lines) <= 5
            
            # Test file info
            info = reader.get_file_info(contract_name)
            assert info['total_lines'] > 0
            assert info['name'] == contract_name
    
    def test_read_large_contract_performance(self):
        """Test performance with large contract files."""
        # This test only runs if contracts directory exists
        contracts_dir = Path("contracts")
        if not contracts_dir.exists():
            pytest.skip("No contracts directory found")
        
        reader = ContractReader()
        contracts = reader.list_contracts()
        
        if contracts:
            # Find the largest contract
            largest_contract = None
            max_size = 0
            
            for contract in contracts[:5]:  # Check first 5 to avoid too much overhead
                try:
                    info = reader.get_file_info(contract)
                    if info['size_bytes'] > max_size:
                        max_size = info['size_bytes']
                        largest_contract = contract
                except:
                    continue
            
            if largest_contract:
                # Test reading with limit should be fast
                result = reader.read(largest_contract, limit=10)
                lines = result.split('\n')
                assert len(lines) <= 10