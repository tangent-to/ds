# Quick Reference Card

## Run Locally

```bash
cd docs
bundle exec jekyll serve
# Visit http://localhost:4000/ds/
```

## File Locations

| What | Where |
|------|-------|
| Site config | `docs/_config.yml` |
| Home page | `docs/index.md` |
| Tutorials | `docs/_tutorials/*.md` |
| API docs | `docs/_api/*.md` |
| Logo | `docs/assets/images/logo.png` |
| Examples | `docs/examples.md` |

## Adding a New Tutorial

1. Create `docs/_tutorials/my-tutorial.md`
2. Add front matter:

```yaml
---
layout: default
title: "My Tutorial"
parent: Tutorials
nav_order: 5
---
```

3. Write content in markdown
4. It auto-appears in navigation!

## Adding to API Reference

1. Create `docs/_api/my-module.md`
2. Add front matter:

```yaml
---
layout: default
title: "My Module"
parent: API Reference
nav_order: 5
---
```

## Common Commands

```bash
# Install/update dependencies
bundle install

# Run with live reload
bundle exec jekyll serve --livereload

# Build for production
bundle exec jekyll build

# Clean build cache
bundle exec jekyll clean
```

## Markdown Tips

```markdown
# Heading 1
## Heading 2

**Bold** and *italic*

[Link text](url)

`inline code`

\`\`\`javascript
// Code block
const x = 1;
\`\`\`

- Bullet list
1. Numbered list

> Blockquote

| Table | Header |
|-------|--------|
| Cell  | Cell   |
```

## Just the Docs Special Features

```markdown
{: .fs-9 }        <!-- Large font -->
{: .fw-300 }      <!-- Light weight -->
{: .btn }         <!-- Button style -->
{: .no_toc }      <!-- Exclude from TOC -->

## Table of Contents
{: .no_toc .text-delta }
1. TOC
{:toc}
```

## Deployment

Push to `main` → Auto-deploys to GitHub Pages!

Check status: https://github.com/tangent-to/ds/actions

## Logo Specs

- File: `docs/assets/images/logo.png`
- Size: 200x200px recommended
- Format: PNG with transparency
- Concept: Possum + triangle (2001: A Space Odyssey style)

Temporarily disable:
```yaml
# logo: "/assets/images/logo.png"
```

## Configuration Highlights

In `docs/_config.yml`:

```yaml
title: "DS"                    # Site title
color_scheme: light            # Theme (light/dark)
search_enabled: true           # Enable search
heading_anchors: true          # Heading links
back_to_top: true             # Back to top button
```

## Need Help?

- Setup: Read `docs/SETUP.md`
- Theme docs: https://just-the-docs.com/
- Jekyll docs: https://jekyllrb.com/docs/
