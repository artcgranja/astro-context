# SKILL.md Loader for astro-context

**Date:** 2026-03-12
**Status:** Approved
**Scope:** Add SKILL.md loading support to astro-context's agent skill system

## Summary

Add a SKILL.md loader that parses the open Agent Skills standard (agentskills.io) into native `Skill` instances, enabling astro-context agents to consume skills from the 2026 cross-platform skill ecosystem alongside existing Python-native skills.

## Context

The SKILL.md format (published by Anthropic, Dec 2025) is the de facto standard for AI coding assistant skills in 2026, adopted by Claude Code, Codex, Copilot, Cursor, Windsurf, and Antigravity. astro-context already has a progressive tool disclosure system (`Skill`, `SkillRegistry`, `@tool` decorator) that aligns naturally with SKILL.md's design.

No other context engineering toolkit bridges development-time SKILL.md files with runtime agent skills. This is a differentiator.

## Design Decisions

1. **Unified registry** -- SKILL.md skills and Python-native skills coexist in the same `SkillRegistry`, same activation model, same API. No subclasses, no type markers.
2. **Hybrid skills** -- A skill can be instructions-only (LLM guidelines) OR instructions + executable Python tools. The loader detects what's available.
3. **Both auto-scan and explicit loading** -- `with_skills_directory()` loads all skills from a path; `with_skill_from_path()` loads one specific skill.
4. **No new dependencies** -- Uses a lightweight regex-based YAML frontmatter parser (split on `---` delimiters, parse key-value pairs). No PyYAML required.
5. **Approach 1 (Loader as utility)** -- Chosen over subclass or plugin system approaches for simplicity.
6. **SKILL.md skills default to `on_demand`** -- Unlike Python-native skills which default to `"always"`, SKILL.md skills default to `"on_demand"` because they come from external sources and should be explicitly activated. The loader sets this default before constructing the `Skill` instance.
7. **Validation in the loader, not the model** -- Name format (lowercase, hyphens, max 64 chars) and description length (max 1024 chars) are validated by the loader only. The `Skill` model stays unchanged (`name: str`, `description: str`) to avoid breaking existing Python-native skills.
8. **`scripts/` directory is out of scope for v1** -- Only `tools.py` in the skill directory is supported. `scripts/` support can be added later.

## Architecture

### SKILL.md to Skill Mapping

| SKILL.md field       | Skill field    | Notes                                                    |
|----------------------|----------------|----------------------------------------------------------|
| `name` (frontmatter) | `name`         | Required. Loader validates: lowercase, hyphens, max 64 chars |
| `description`        | `description`  | Required. Loader validates: max 1024 chars               |
| Markdown body        | `instructions` | Full markdown content after frontmatter                  |
| `tools.py`           | `tools`        | Auto-discovered `@tool`-decorated functions (see Tool Discovery) |
| `activation`         | `activation`   | Loader defaults to `"on_demand"` (not model default `"always"`), overridable in frontmatter |
| `tags`               | `tags`         | Passed through if present as YAML list                   |

### New Module: `agent/skills/loader.py`

The loader performs three steps:

1. **Parse** -- Read SKILL.md, split on `---` delimiters to extract frontmatter (key-value pairs) and markdown body. No PyYAML dependency.
2. **Discover tools** -- If `tools.py` exists in the skill directory, import it using `importlib.util.spec_from_file_location` + `module_from_spec` + `exec_module`. Scan module-level attributes for `AgentTool` instances (the `@tool` decorator returns `AgentTool`). The imported module is added to `sys.modules` with a namespaced key (`astro_context.skills.<skill_name>.tools`) to avoid collisions.
3. **Build Skill** -- Gather all tools into a tuple, then create the frozen `Skill(...)` instance in a single call (no mutation after construction).

**Tool Discovery Details:**
- Only `tools.py` in the skill root directory is scanned (not `scripts/`).
- The loader iterates over `dir(module)` and collects any attribute that is an `AgentTool` instance.
- Import errors in `tools.py` raise `ValueError` with a message indicating the skill name and the underlying error.
- If `tools.py` does not exist, the skill is instructions-only (empty tools tuple). This is valid.

Key functions:

```python
def load_skill(path: Path) -> Skill:
    """Load a single SKILL.md directory into a Skill instance.

    Raises:
        FileNotFoundError: If path or SKILL.md does not exist.
        ValueError: If frontmatter is invalid (missing name/description,
                    name format violation, description too long) or if
                    tools.py import fails.
    """

def load_skills_directory(path: Path) -> list[Skill]:
    """Scan a directory for */SKILL.md patterns, return all loaded skills.

    Raises:
        FileNotFoundError: If the directory does not exist.

    Returns an empty list (with a warning log) if no SKILL.md files are found.
    Logs and skips individual skills that fail to load (does not abort the batch).
    """
```

**Path Resolution:** Relative paths are resolved relative to CWD (`Path(path).resolve()`). This applies to both `load_skill` and `load_skills_directory`, as well as the Agent convenience methods.

### SkillRegistry Extensions

Two new methods on `SkillRegistry`. No new instance attributes -- these are stateless wrappers that call the loader and then `self.register()`. The `__slots__` tuple is unchanged.

If a SKILL.md skill has the same name as an already-registered skill, `register()` raises `ValueError` (existing behavior). `load_from_directory` logs the error and skips the duplicate, continuing with remaining skills.

```python
def load_from_path(self, path: Path) -> Skill:
    """Load a SKILL.md skill and register it. Returns the skill."""

def load_from_directory(self, path: Path) -> list[Skill]:
    """Load all SKILL.md skills from a directory and register them."""
```

No changes to existing methods (`register`, `activate`, `active_tools`, `skill_discovery_prompt`). The `activate_skill` meta-tool already handles injecting `skill.instructions` into the conversation when activated -- this works identically for SKILL.md-loaded skills since they are plain `Skill` instances.

### Agent Convenience Methods

Two new methods on `Agent`. No new instance attributes needed -- both methods delegate to `self._skill_registry` and call `self._ensure_activate_tool()`. The `__slots__` tuple is unchanged.

```python
def with_skills_directory(self, path: str | Path) -> Agent:
    """Load all SKILL.md skills from a directory. Returns self for chaining."""

def with_skill_from_path(self, path: str | Path) -> Agent:
    """Load one SKILL.md skill from a directory. Returns self for chaining."""
```

Usage:

```python
agent = (
    Agent(model="claude-sonnet-4-5-20251001")
    .with_system_prompt("You are a helpful assistant.")
    .with_skills_directory("./skills")           # all SKILL.md skills
    .with_skill_from_path("./extras/brainstorm") # one specific skill
    .with_skill(memory_skill(memory))            # native Python skill
)
```

### Security Considerations

- Only load from explicitly configured directories (not arbitrary URLs)
- Log what Python modules get imported during tool discovery
- Tool discovery uses controlled `importlib` imports, not `exec`

## Test Brainstorming Skill

A hybrid SKILL.md skill that validates both loader paths:

```
examples/skills/brainstorm/
    SKILL.md     # Instructions for brainstorming flow
    tools.py     # save_brainstorm_result tool
```

**SKILL.md frontmatter:**
- `name: brainstorm`
- `description: Guide the agent through a structured brainstorming process`
- `activation: on_demand`
- `tags: [creative]`

**SKILL.md body:**
Instructions guiding the agent through: understand goal, ask clarifying questions, propose 2-3 approaches with trade-offs, summarize chosen approach.

**tools.py:**
- `save_brainstorm_result(title: str, summary: str, approaches: str) -> str` -- Persists brainstorm output to a JSON file.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `src/astro_context/agent/skills/loader.py` | SKILL.md parser + tool discovery |
| `tests/agent/skills/test_loader.py` | Unit tests for loader |
| `tests/agent/skills/test_loader_integration.py` | Integration tests |
| `tests/fixtures/skills/brainstorm/SKILL.md` | Test brainstorming skill |
| `tests/fixtures/skills/brainstorm/tools.py` | Test tool (save_brainstorm_result) |
| `tests/fixtures/skills/minimal/SKILL.md` | Instructions-only test skill |
| `tests/fixtures/skills/invalid/SKILL.md` | Invalid frontmatter test case |
| `examples/skills/brainstorm/SKILL.md` | Example brainstorming skill |
| `examples/skills/brainstorm/tools.py` | Example tool implementation |

### Modified Files

| File | Change |
|------|--------|
| `src/astro_context/agent/skills/registry.py` | Add `load_from_path()`, `load_from_directory()` |
| `src/astro_context/agent/agent.py` | Add `with_skills_directory()`, `with_skill_from_path()` |
| `src/astro_context/agent/skills/__init__.py` | Export loader functions |
| `src/astro_context/agent/__init__.py` | Export loader functions |

## Testing Strategy

### Unit Tests (`tests/agent/skills/test_loader.py`)

- Parse SKILL.md with valid frontmatter -> correct Skill fields
- Parse SKILL.md without tools -> instructions-only skill (empty tools tuple)
- Parse SKILL.md with tools.py -> discovers @tool-decorated functions
- Invalid frontmatter (missing name/description) -> raises `ValueError`
- Invalid name format (uppercase, special chars, too long) -> raises `ValueError`
- Description exceeding 1024 chars -> raises `ValueError`
- Nonexistent SKILL.md path -> raises `FileNotFoundError`
- `tools.py` with import error -> raises `ValueError` with context
- `load_skills_directory` -> finds all */SKILL.md in directory
- `load_skills_directory` with nonexistent dir -> raises `FileNotFoundError`
- `load_skills_directory` with empty dir -> returns empty list (logs warning)
- `load_skills_directory` with one invalid skill -> skips it, loads the rest

### Integration Tests (`tests/agent/skills/test_loader_integration.py`)

- Load brainstorm skill from SKILL.md -> registers in SkillRegistry
- Activate brainstorm skill -> tools become available via active_tools()
- Execute save_brainstorm_result tool -> produces output
- Agent.with_skills_directory() -> loads and registers all skills
- Agent.with_skill_from_path() -> loads one specific skill
- Mix of native Python skill + SKILL.md skill in same registry -> no conflicts
- Duplicate skill name (SKILL.md vs native) -> raises ValueError on register
- `load_from_directory` with duplicate -> skips duplicate, loads rest

### Test Fixtures

Directory `tests/fixtures/skills/` with sample skills: valid, invalid, instructions-only, with-tools.

## Out of Scope

- Plugin system with versioning/dependency resolution (future)
- Skills marketplace or registry service (future)
- Separate skills repository with diverse skills (future, after loader is validated)
- Full SKILL.md spec support (assets/, references/, scripts/) -- start with frontmatter + markdown + tools.py only
