# tangent/ds documentation Site

This directory contains the Jekyll-based documentation site for tangent/ds.

## Local development

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Setup

```bash
cd docs
bundle install
```

### Run locally

```bash
bundle exec jekyll serve
```

Then visit `http://localhost:4000/ds/`

### Live reload

For automatic rebuilding on file changes:

```bash
bundle exec jekyll serve --livereload
```

## Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── Gemfile                  # Ruby dependencies
├── index.md                 # Home page
├── getting-started.md       # Getting started guide
├── tutorials.md             # Tutorials index
├── api-reference.md         # API reference index
├── examples.md              # Code examples
├── _tutorials/              # Tutorial content (to be added)
├── _api/                    # API documentation (to be added)
└── assets/
    └── images/
        └── logo.png         # Site logo (placeholder)
```

## Adding content

### New tutorial

1. Create a markdown file in `_tutorials/`
2. Add front matter:

```yaml
---
layout: default
title: "Your Tutorial Title"
parent: Tutorials
nav_order: 1
---
```

### New API documentation

1. Create a markdown file in `_api/`
2. Add front matter:

```yaml
---
layout: default
title: "Module Name"
parent: API Reference
nav_order: 1
---
```

## Theme

This site uses [Just the Docs](https://just-the-docs.com/) theme with minimal customization.

## Deployment

The site is configured to deploy to GitHub Pages automatically when changes are pushed to the main branch.

GitHub Pages URL: `https://tangent-to.github.io/ds/`
