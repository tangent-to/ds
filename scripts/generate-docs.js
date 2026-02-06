#!/usr/bin/env node
/**
 * Generate API documentation from source code
 * Creates markdown files in the docs/api/ directory with proper Jekyll front matter
 *
 * Usage: node scripts/generate-docs.js
 *
 * NOTE: This script generates skeletal API docs from exports and JSDoc comments.
 * The hand-written docs in docs/api/ are more comprehensive and should be
 * preferred. Run this script only to scaffold new modules.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');
const apiDocsDir = path.join(rootDir, 'docs', 'api');

// Module metadata for generating proper front matter
const MODULE_META = {
  core: { title: 'Core Utilities', navOrder: 5, description: 'Math, linear algebra, data manipulation, formula parsing' },
  stats: { title: 'Statistics', navOrder: 1, description: 'GLM, hypothesis tests, distributions, model comparison' },
  ml: { title: 'Machine Learning', navOrder: 2, description: 'KNN, trees, forests, MLP, preprocessing, validation' },
  mva: { title: 'Multivariate Analysis', navOrder: 3, description: 'PCA, LDA, RDA, CCA ordination methods' },
  plot: { title: 'Visualization', navOrder: 4, description: 'Observable Plot configs for biplots, ROC, diagnostics' },
};

// Create docs/api directory
if (!fs.existsSync(apiDocsDir)) {
  fs.mkdirSync(apiDocsDir, { recursive: true });
}

/**
 * Extract JSDoc comments paired with their following export
 */
function extractDocumentedExports(content) {
  const results = [];

  // Match JSDoc comment followed by export
  const pattern = /\/\*\*([\s\S]*?)\*\/\s*\n\s*(export\s+(?:class|function|const|let|var)\s+(\w+))/g;
  let match;

  while ((match = pattern.exec(content)) !== null) {
    const comment = match[1]
      .split('\n')
      .map(line => line.trim().replace(/^\*\s?/, ''))
      .join('\n')
      .trim();

    results.push({
      name: match[3],
      type: match[2].match(/export\s+(class|function|const|let|var)/)?.[1] || 'named',
      description: comment.split('\n')[0], // First line as summary
      fullDoc: comment,
    });
  }

  // Also match named exports: export { name } from './file.js'
  const namedPattern = /export\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]/g;
  while ((match = namedPattern.exec(content)) !== null) {
    const names = match[1].split(',').map(n => n.trim().split(/\s+as\s+/)[0].trim());
    const source = match[2];
    for (const name of names) {
      if (name && !results.find(r => r.name === name)) {
        results.push({ name, type: 'named', description: `From ${source}`, fullDoc: '' });
      }
    }
  }

  return results;
}

/**
 * Generate a markdown page for a module (only if it doesn't already exist)
 */
function generateModulePage(moduleName, indexPath) {
  const meta = MODULE_META[moduleName];
  if (!meta) return null;

  const slug = meta.title.toLowerCase().replace(/\s+/g, '-');
  const outputPath = path.join(apiDocsDir, `${slug === 'statistics' ? 'statistics' : slug === 'machine-learning' ? 'machine-learning' : slug === 'multivariate-analysis' ? 'multivariate' : slug === 'visualization' ? 'visualization' : slug}.md`);

  // Skip if hand-written docs already exist
  if (fs.existsSync(outputPath)) {
    console.log(`  Skipping ${moduleName} (hand-written docs exist at ${path.basename(outputPath)})`);
    return outputPath;
  }

  const content = fs.readFileSync(indexPath, 'utf8');
  const exports = extractDocumentedExports(content);

  let md = `---
layout: default
title: ${meta.title}
parent: API Reference
nav_order: ${meta.navOrder}
permalink: /api/${path.basename(outputPath, '.md')}
---

# ${meta.title} API
{: .no_toc }

${meta.description}.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Exports

`;

  for (const exp of exports) {
    md += `### \`${exp.name}\`\n\n`;
    if (exp.description) {
      md += `${exp.description}\n\n`;
    }
    md += `**Type:** ${exp.type}\n\n---\n\n`;
  }

  fs.writeFileSync(outputPath, md);
  return outputPath;
}

/**
 * Main documentation generator
 */
function generateDocs() {
  console.log('Generating API documentation scaffolds...\n');

  const modules = fs.readdirSync(srcDir)
    .filter(name => {
      const fullPath = path.join(srcDir, name);
      return fs.statSync(fullPath).isDirectory() && MODULE_META[name];
    });

  for (const moduleName of modules) {
    const indexPath = path.join(srcDir, moduleName, 'index.js');

    if (fs.existsSync(indexPath)) {
      console.log(`Processing ${moduleName}...`);
      generateModulePage(moduleName, indexPath);
    }
  }

  console.log(`\nDone. API docs are in docs/api/`);
  console.log('Note: Hand-written docs are preserved; only missing modules are scaffolded.');
}

// Run
generateDocs();
