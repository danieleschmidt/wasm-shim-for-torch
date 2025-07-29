"""Tests for CLI functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch
from wasm_torch.cli import main, download_runtime, export_model


def test_download_runtime_not_implemented():
    """Test that download_runtime raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        download_runtime("latest", Path("/tmp"))


def test_export_model_not_implemented():
    """Test that export_model raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        export_model(Path("model.pt"), Path("model.wasm"), "O2")


@patch('sys.argv', ['wasm-torch', '--help'])
def test_cli_help():
    """Test CLI help doesn't crash."""
    with pytest.raises(SystemExit):
        main()


@patch('sys.argv', ['wasm-torch'])
def test_cli_no_command():
    """Test CLI with no command."""
    with pytest.raises(SystemExit):
        main()


@patch('wasm_torch.cli.download_runtime')
@patch('sys.argv', ['wasm-torch', 'download-runtime', '--version', 'latest'])
def test_cli_download_command(mock_download):
    """Test download command parsing."""
    mock_download.side_effect = NotImplementedError()
    
    with pytest.raises(NotImplementedError):
        main()
    
    mock_download.assert_called_once()


@patch('wasm_torch.cli.export_model')
@patch('sys.argv', ['wasm-torch', 'export', 'model.pt', '--output', 'model.wasm'])
def test_cli_export_command(mock_export):
    """Test export command parsing."""
    mock_export.side_effect = NotImplementedError()
    
    with pytest.raises(NotImplementedError):
        main()
    
    mock_export.assert_called_once()