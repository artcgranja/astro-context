# Anchor Rebrand & Docs Visual Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename anchor → anchor everywhere (code, docs, package, CLI) and apply Astro brand visual identity to the documentation site.

**Architecture:** Two-phase approach — (1) bulk rename via sed/mv commands for the mechanical rename, (2) manual precision updates for visual theming, icon, and brand-specific changes that sed can't handle.

**Tech Stack:** MkDocs Material, Python 3.11+, sed, find, Mermaid diagrams

---

## Context & Research Summary

### Brand Colors (from ~/github/astro-webpage)

| Token | Hex | Usage |
|-------|-----|-------|
| Primary Blue | `#3b82f6` | Buttons, links, primary accent |
| Primary Blue Dim | `#2563eb` | Hover states |
| Dark Void | `#020810` | Darkest background |
| Dark Void Light | `#0a1420` | Dark mode surfaces |
| Dark Void Lighter | `#101c2c` | Dark mode elevated surfaces |
| Light Void | `#FAFAF8` | Light mode background |
| Light Void Light | `#ECEEE9` | Light mode surfaces |
| Steel | `#64748b` | Secondary accent |
| Steel Hover | `#94a3b8` | Secondary hover |
| Sage Green | `#6B8E6B` | Organic accent |
| Text Primary (dark) | `#e8ecf0` | Body text on dark |
| Text Dim (dark) | `#8896a4` | Secondary text on dark |
| Text Primary (light) | `#0F172A` | Body text on light |
| Icon Blue | `#2563eb` | Icon color dark mode |
| Icon Blue (light) | `#1D4ED8` | Icon color light mode |

### Brand Fonts

| Role | Font | Fallback |
|------|------|----------|
| Display | Space Grotesk (400-700) | system-ui, sans-serif |
| Body | system-ui | -apple-system, BlinkMacSystemFont, sans-serif |
| Code | Fira Code (400, 500) | JetBrains Mono, monospace |

### Visual Best Practices (2026 Research)

From analyzing LangChain, Agno, Pydantic, FastAPI, Polars docs:

1. **Dark-first theming** — all top libraries support it, most default to it
2. **Single bold accent color** on neutral base — creates strong brand identity
3. **Card-based landing pages** with clear 3-path funnel (quickstart, tutorial, API)
4. **Tabbed, copyable code blocks** — minimum bar
5. **Interactive hover effects** — card lifts, color transitions (Agno-style)
6. **Social proof** — badges, enterprise logos, contributor avatars
7. **Mermaid diagrams** for architecture — replacing ASCII art
8. **Space Grotesk / Inter / system-ui** trending for body text

### Icon for "Anchor"

**Recommendation:** Use Material icon `material/anchor` (⚓). It's:
- Available in Material for MkDocs icon set
- Literally matches the name
- Conveys stability, grounding, connection — fits a "context engineering" toolkit
- Works at all sizes (favicon, nav, badges)

Alternative: Custom SVG anchor icon if we want unique branding.

---

## Rename Scope Analysis

| Area | Occurrences | Method |
|------|-------------|--------|
| Python source (`anchor`) | ~738 in 209 files | `sed` bulk replace |
| Python source (`anchor`) | ~67 in 32 files | `sed` bulk replace |
| Docs markdown (`anchor`) | ~374 in 48 files | `sed` bulk replace |
| Docs markdown (`anchor`) | ~167 in 43 files | `sed` bulk replace |
| pyproject.toml | 11 occurrences | `sed` + manual review |
| mkdocs.yml | 6 occurrences | `sed` + manual review |
| CI/CD (.github/workflows/) | 1 occurrence | `sed` |
| HTML/CSS (docs) | 2 occurrences | `sed` |
| llms.txt | 12 occurrences | `sed` |
| **Directory rename** | `src/anchor/` → `src/anchor/` | `mv` |

**Do NOT rename:**
- Author name "Astro" in pyproject.toml (company name)
- Copyright "Astro" in mkdocs.yml (company name)
- Email `astro@anchor.dev` → needs manual update

---

## Phase 1: Bulk Rename (anchor → anchor)

### Task 1: Rename source directory

**Files:**
- Rename: `src/anchor/` → `src/anchor/`

- [ ] **Step 1: Rename the directory**
```bash
mv src/anchor src/anchor
```

- [ ] **Step 2: Verify directory structure**
```bash
ls src/anchor/
```
Expected: Same contents as before (agent/, cache/, evaluation/, etc.)

---

### Task 2: Bulk sed replace across all files

**Files:** All .py, .md, .toml, .yml, .yaml, .txt, .css, .html, .json, .cfg files

- [ ] **Step 1: Replace underscore variant (Python imports)**
```bash
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" -o -name "*.txt" -o -name "*.cfg" -o -name "*.css" -o -name "*.html" -o -name "*.json" \) \
  ! -path "./.git/*" ! -path "*/node_modules/*" ! -path "*/site/*" ! -path "*uv.lock*" \
  -exec sed -i '' 's/anchor/anchor/g' {} +
```

- [ ] **Step 2: Replace hyphen variant (package name, URLs, docs)**
```bash
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" -o -name "*.txt" -o -name "*.cfg" -o -name "*.css" -o -name "*.html" -o -name "*.json" \) \
  ! -path "./.git/*" ! -path "*/node_modules/*" ! -path "*/site/*" ! -path "*uv.lock*" \
  -exec sed -i '' 's/anchor/anchor/g' {} +
```

- [ ] **Step 3: Verify no remaining references**
```bash
grep -r "astro.context" --include="*.py" --include="*.md" --include="*.toml" --include="*.yml" --include="*.txt" --include="*.css" --include="*.html" . | grep -v ".git/" | grep -v "site/" | grep -v "uv.lock"
```
Expected: Zero results (or only "Astro" company name references)

---

### Task 3: Manual precision fixes

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/mkdocs.yml`
- Modify: `docs/overrides/main.html`

- [ ] **Step 1: Fix pyproject.toml**
After sed, verify these are correct:
- Package name: `name = "anchor"`
- CLI entry: `anchor = "anchor.cli:app"`
- GitHub URLs: `github.com/arthurgranja/anchor`
- Author email: Update if needed
- `known-first-party = ["anchor"]`
- `source = ["anchor"]`

- [ ] **Step 2: Fix mkdocs.yml**
Verify:
- `site_name: anchor`
- `site_url: https://arthurgranja.github.io/anchor/`
- `repo_url: https://github.com/arthurgranja/anchor`
- `repo_name: arthurgranja/anchor`
- `copyright: Copyright &copy; 2026 Astro — MIT License` (keep "Astro")
- magiclink `repo: anchor`

- [ ] **Step 3: Fix announcement bar**
```html
Star us on GitHub — anchor v0.1.0 is here!
```

- [ ] **Step 4: Verify build**
```bash
cd docs && uv run mkdocs build --strict 2>&1
```
Expected: Clean build, no warnings

- [ ] **Step 5: Run tests**
```bash
uv run pytest --tb=short -q 2>&1 | tail -20
```
Expected: All tests pass

- [ ] **Step 6: Commit**
```bash
git add -A
git commit -m "refactor: rename anchor to anchor across entire codebase"
```

---

## Phase 2: Visual Rebrand (Astro Brand Identity)

### Task 4: Update mkdocs.yml theme

**Files:**
- Modify: `docs/mkdocs.yml` (theme section)

- [ ] **Step 1: Update palette and fonts**

Change:
```yaml
theme:
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: blue          # was: deep purple
      accent: teal           # was: amber (closest Material preset to our steel blue)
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: blue          # was: deep purple
      accent: teal           # was: amber
  font:
    text: ""                 # empty = system font stack
    code: Fira Code          # was: JetBrains Mono
  icon:
    logo: material/anchor    # anchor icon!
```

- [ ] **Step 2: Add favicon**
Add to theme section:
```yaml
  favicon: assets/anchor-favicon.png
```
(We'll create this in Task 6)

---

### Task 5: Rewrite extra.css with Astro brand

**Files:**
- Rewrite: `docs/docs/stylesheets/extra.css`

- [ ] **Step 1: Write new CSS**

Key changes:
- Hero gradient: `#020810` → `#0a1420` (dark) / `#FAFAF8` → `#ECEEE9` (light)
- Primary accent: `#3b82f6` (replaces `#7c4dff`)
- Card hover shadows: `rgba(59, 130, 246, 0.10)` (replaces purple)
- Code block border: `#3b82f6`
- Add Space Grotesk import for headings
- Add Fira Code import for code
- Glass effects from astro-webpage (subtle `backdrop-filter: blur()`)
- Dark mode overrides using `[data-md-color-scheme="slate"]`
- Smooth card hover transitions (lift + blue glow)

CSS structure:
```
/* ── Fonts ── */
@import url(Space Grotesk + Fira Code from Google Fonts)

/* ── CSS Custom Properties ── */
:root { --anchor-primary: #3b82f6; ... }
[data-md-color-scheme="slate"] { --anchor-bg: #020810; ... }

/* ── Hero Section ── */
.hero { gradient background, large text }

/* ── Feature Cards ── */
.grid.cards .card { hover lift + blue shadow }

/* ── Code Blocks ── */
.highlight { left border accent, dark bg }

/* ── Typography ── */
h1, h2, h3 { font-family: Space Grotesk }

/* ── Responsive ── */
@media (max-width: 768px) { ... }
```

---

### Task 6: Update Mermaid diagram colors

**Files:**
- Modify: `docs/docs/index.md` (4 style lines)
- Review: `docs/docs/concepts/architecture.md` (keep as-is, colors are semantic not brand)

- [ ] **Step 1: Update index.md Mermaid colors**

Replace:
- `#7c4dff` → `#3b82f6` (primary blue)
- `#ffab00` → `#6B8E6B` (sage green for formatter)

- [ ] **Step 2: Leave architecture.md as-is**
The architecture diagram uses semantic colors (green for collection, orange for processing, etc.) — these are intentional and don't need brand alignment.

---

### Task 7: Update landing page content for "anchor"

**Files:**
- Modify: `docs/docs/index.md`

- [ ] **Step 1: Update tagline and branding**
- Title: `# anchor`
- Tagline: keep "Context is the product. The LLM is just the consumer." (good as-is)
- Badges: update PyPI/GitHub URLs to anchor
- Comparison table: "anchor" column header

- [ ] **Step 2: Add anchor icon to hero**
Use Material icon in hero section:
```markdown
:material-anchor:{ .hero-icon }
```

---

### Task 8: Create favicon and logo assets

**Files:**
- Create: `docs/docs/assets/anchor-favicon.png`

- [ ] **Step 1: Generate simple anchor favicon**
Use a simple SVG → PNG conversion or reference Material icon.
For now, just set `logo: material/anchor` in mkdocs.yml (built-in icon).

---

### Task 9: Final verification and commit

- [ ] **Step 1: Build docs**
```bash
cd docs && uv run mkdocs build --strict 2>&1
```
Expected: Clean build

- [ ] **Step 2: Serve and visual check**
```bash
cd docs && uv run mkdocs serve --dev-addr 127.0.0.1:8000
```
Check: Landing page, dark mode, card hovers, code blocks, Mermaid diagrams

- [ ] **Step 3: Run full test suite**
```bash
uv run pytest --tb=short -q
```
Expected: All 1988 tests pass

- [ ] **Step 4: Commit visual changes**
```bash
git add -A
git commit -m "feat: apply Astro brand identity to anchor docs (colors, fonts, icon)"
```

---

## Post-Implementation Notes

### GitHub Repo Rename
The user must manually rename the repo:
1. Go to `github.com/arthurgranja/anchor` → Settings → Repository name
2. Change to `anchor`
3. GitHub auto-redirects old URLs

### PyPI
If publishing to PyPI, the package name `anchor` may already be taken. Check:
```bash
pip index versions anchor 2>/dev/null
```
Alternative names if taken: `anchor-context`, `anchor-ai`, `anchorlib`

### What Stays "Astro"
- Author name in pyproject.toml: "Astro" (company)
- Copyright in mkdocs.yml: "Astro" (company)
- The company/org name is separate from the library name
