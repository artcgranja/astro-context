"""Tests for astro_context.cli.

Exercises the Typer CLI app via CliRunner, covering the version flag,
info command, index command (success and error paths), and query command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from astro_context import __version__
from astro_context.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# main callback (--version)
# ---------------------------------------------------------------------------


class TestMainCallback:
    """The root callback with --version flag.

    The callback does not set ``invoke_without_command=True``, so
    ``--version`` must be paired with a subcommand for the callback to
    execute.  Invoking with no subcommand results in exit code 2.
    """

    def test_version_flag_prints_version_and_exits(self) -> None:
        """--version before a subcommand prints the version and exits."""
        result = runner.invoke(app, ["--version", "info"])
        assert result.exit_code == 0
        assert "astro-context" in result.output
        assert __version__ in result.output

    def test_short_version_flag(self) -> None:
        """-v is accepted as the short form of --version."""
        result = runner.invoke(app, ["-v", "info"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_subcommand_exits_with_error(self) -> None:
        """Invoking without a subcommand fails with 'Missing command'."""
        result = runner.invoke(app, [])
        assert result.exit_code == 2
        assert "Missing command" in result.output

    def test_help_flag(self) -> None:
        """--help shows usage information and available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Context engineering toolkit" in result.output
        assert "info" in result.output
        assert "index" in result.output
        assert "query" in result.output


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestInfoCommand:
    """The ``info`` subcommand."""

    def test_info_runs_successfully(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0

    def test_info_shows_version(self) -> None:
        result = runner.invoke(app, ["info"])
        assert __version__ in result.output

    def test_info_shows_python_version(self) -> None:
        import sys

        python_version = sys.version.split()[0]
        result = runner.invoke(app, ["info"])
        assert python_version in result.output

    def test_info_shows_dependency_names(self) -> None:
        result = runner.invoke(app, ["info"])
        # At least the dependency names should appear in the output table
        assert "pydantic" in result.output

    def test_info_shows_missing_dep_when_import_fails(self) -> None:
        """When a dependency cannot be imported, the table shows 'not installed'."""
        with patch("importlib.import_module", side_effect=ImportError("mocked")):
            result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "not installed" in result.output


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


class TestIndexCommandWithFile:
    """The ``index`` subcommand on a single file."""

    def test_index_valid_text_file(self, tmp_path: Path) -> None:
        """Indexing a small UTF-8 text file succeeds and reports token count."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello world, this is a test.", encoding="utf-8")

        result = runner.invoke(app, ["index", str(test_file)])
        assert result.exit_code == 0
        assert "Indexing from" in result.output
        assert "sample.txt" in result.output
        assert "tokens" in result.output

    def test_index_with_custom_chunk_size(self, tmp_path: Path) -> None:
        """The --chunk-size option is accepted and echoed."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("Some content here.", encoding="utf-8")

        result = runner.invoke(app, ["index", str(test_file), "--chunk-size", "256"])
        assert result.exit_code == 0
        assert "chunk_size=256" in result.output

    def test_index_with_short_chunk_size_flag(self, tmp_path: Path) -> None:
        """The -c short option works the same as --chunk-size."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("Some content here.", encoding="utf-8")

        result = runner.invoke(app, ["index", str(test_file), "-c", "128"])
        assert result.exit_code == 0
        assert "chunk_size=128" in result.output


class TestIndexCommandNonExistentPath:
    """The ``index`` subcommand when the path does not exist."""

    def test_index_nonexistent_path_exits_with_error(self) -> None:
        result = runner.invoke(app, ["index", "/no/such/path/does_not_exist.txt"])
        assert result.exit_code == 1
        assert "does not exist" in result.output


class TestIndexCommandFileTooLarge:
    """The ``index`` subcommand when a file exceeds _MAX_FILE_SIZE."""

    def test_index_file_too_large(self, tmp_path: Path) -> None:
        """A file whose stat().st_size exceeds the limit triggers an error."""
        large_file = tmp_path / "huge.txt"
        large_file.write_text("x", encoding="utf-8")

        # Mock _MAX_FILE_SIZE to 0 so any file is "too large"
        with patch("astro_context.cli._MAX_FILE_SIZE", 0):
            result = runner.invoke(app, ["index", str(large_file)])

        assert result.exit_code == 1
        assert "too large" in result.output

    def test_index_file_at_exact_limit_succeeds(self, tmp_path: Path) -> None:
        """A file exactly at the limit (not exceeding) is accepted."""
        content = "small"
        test_file = tmp_path / "exact.txt"
        test_file.write_text(content, encoding="utf-8")
        file_size = test_file.stat().st_size

        # Set limit to exactly the file size -- the check is >, not >=
        with patch("astro_context.cli._MAX_FILE_SIZE", file_size):
            result = runner.invoke(app, ["index", str(test_file)])

        assert result.exit_code == 0
        assert "tokens" in result.output


class TestIndexCommandNonUtf8:
    """The ``index`` subcommand with a non-UTF-8 file."""

    def test_index_binary_file_exits_with_error(self, tmp_path: Path) -> None:
        """A file containing invalid UTF-8 bytes triggers a UnicodeDecodeError."""
        bad_file = tmp_path / "binary.txt"
        bad_file.write_bytes(b"\xff\xfe\x00\x01\x80\x81\x82")

        result = runner.invoke(app, ["index", str(bad_file)])
        assert result.exit_code == 1
        assert "not valid UTF-8" in result.output


class TestIndexCommandOSError:
    """The ``index`` subcommand when an OSError occurs during read."""

    def test_index_os_error(self, tmp_path: Path) -> None:
        """An OSError during file reading is caught and reported."""
        test_file = tmp_path / "readable.txt"
        test_file.write_text("content", encoding="utf-8")

        with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            result = runner.invoke(app, ["index", str(test_file)])

        assert result.exit_code == 1
        assert "Error reading" in result.output


class TestIndexCommandDirectory:
    """The ``index`` subcommand on a directory."""

    def test_index_directory_with_text_files(self, tmp_path: Path) -> None:
        """Indexing a directory reports the number of text files found."""
        (tmp_path / "doc1.txt").write_text("First doc.", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Second doc.", encoding="utf-8")
        (tmp_path / "notes.md").write_text("# Notes", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")  # not a text file

        result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "Found 3 text files" in result.output

    def test_index_empty_directory(self, tmp_path: Path) -> None:
        """An empty directory reports 0 text files."""
        result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "Found 0 text files" in result.output

    def test_index_directory_ignores_non_text_files(self, tmp_path: Path) -> None:
        """Only .txt and .md files are counted, not .py or .json."""
        (tmp_path / "code.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")

        result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "Found 1 text files" in result.output

    def test_index_directory_recursive(self, tmp_path: Path) -> None:
        """The glob pattern **/*.txt finds files in nested subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file.", encoding="utf-8")
        (tmp_path / "top.txt").write_text("Top file.", encoding="utf-8")

        result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "Found 2 text files" in result.output

    def test_index_directory_deeply_nested(self, tmp_path: Path) -> None:
        """Files nested several levels deep are discovered by **/*.txt."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("Deep content.", encoding="utf-8")

        result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "Found 1 text files" in result.output


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------


class TestQueryCommand:
    """The ``query`` subcommand (placeholder)."""

    def test_query_basic(self) -> None:
        """The query command runs and echoes the query text."""
        result = runner.invoke(app, ["query", "What is context engineering?"])
        assert result.exit_code == 0
        assert "What is context engineering?" in result.output

    def test_query_with_max_tokens(self) -> None:
        """The --max-tokens option is accepted and echoed."""
        result = runner.invoke(app, ["query", "test query", "--max-tokens", "2048"])
        assert result.exit_code == 0
        assert "2048" in result.output

    def test_query_with_short_max_tokens_flag(self) -> None:
        """The -t short flag works for --max-tokens."""
        result = runner.invoke(app, ["query", "test query", "-t", "1024"])
        assert result.exit_code == 0
        assert "1024" in result.output

    def test_query_with_format_option(self) -> None:
        """The --format option is accepted and echoed."""
        result = runner.invoke(app, ["query", "test query", "--format", "anthropic"])
        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_query_with_short_format_flag(self) -> None:
        """The -f short flag works for --format."""
        result = runner.invoke(app, ["query", "test query", "-f", "openai"])
        assert result.exit_code == 0
        assert "openai" in result.output

    def test_query_with_all_options(self) -> None:
        """All options can be combined in a single invocation."""
        result = runner.invoke(
            app,
            ["query", "combined test", "-t", "512", "-f", "generic"],
        )
        assert result.exit_code == 0
        assert "combined test" in result.output
        assert "512" in result.output
        assert "generic" in result.output

    def test_query_shows_placeholder_note(self) -> None:
        """The placeholder note about indexed documents is shown."""
        result = runner.invoke(app, ["query", "anything"])
        assert result.exit_code == 0
        assert "Requires indexed documents" in result.output
