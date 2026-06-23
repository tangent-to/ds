#!/usr/bin/env node
/**
 * Build the API reference for the Jekyll (just-the-docs) site from JSDoc.
 *
 * Pipeline:
 *   1. Run TypeDoc (config in typedoc.json) -> raw markdown in docs/api-generated/
 *   2. Post-process each page into a just-the-docs-friendly page:
 *        - add Jekyll front matter (layout/title/parent/grand_parent/nav_order/permalink)
 *        - rewrite relative `*.md` links to the target page's permalink
 *   3. Write the result into docs/api/ (replacing the previous auto-built pages)
 *
 * The nav is three levels deep, which is the just-the-docs maximum:
 *   API Reference  ->  Module (core/stats/ml/mva/plot)  ->  Namespace
 *
 * Usage: node scripts/build-api-docs.mjs   (or: npm run docs:api)
 */

import { execSync } from 'node:child_process';
import {
  rmSync, mkdirSync, readdirSync, readFileSync, writeFileSync, statSync,
} from 'node:fs';
import { join, dirname, relative, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RAW = join(ROOT, 'docs', 'api-generated');
const OUT = join(ROOT, 'docs', 'api');

// Module display metadata. `slug` matches the historical hand-written page
// permalinks so existing links (footer, external) keep working.
const MODULES = {
  stats: { title: 'Statistics', navOrder: 1, slug: 'statistics' },
  ml: { title: 'Machine Learning', navOrder: 2, slug: 'machine-learning' },
  mva: { title: 'Multivariate Analysis', navOrder: 3, slug: 'multivariate' },
  plot: { title: 'Visualization', navOrder: 4, slug: 'visualization' },
  core: { title: 'Core Utilities', navOrder: 5, slug: 'core' },
};

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (name.endsWith('.md')) out.push(p);
  }
  return out;
}

// Classify a generated file (path relative to RAW, posix) into a site page.
// Returns { module, ns | null, permalink, title, isModuleIndex } or null to drop.
function classify(relPath) {
  const parts = relPath.split('/');
  const module = parts[0].replace(/\.md$/, '');
  if (!MODULES[module]) return null; // drop the root index.md and anything unknown

  const mslug = MODULES[module].slug;

  // Module landing page: "<module>/index.md" or single-file "<module>.md".
  const isModuleIndex = relPath === `${module}/index.md` || relPath === `${module}.md`;
  if (isModuleIndex) {
    return {
      module,
      ns: null,
      permalink: `/api/${mslug}`,
      title: MODULES[module].title,
      fileSlug: mslug,
      isModuleIndex: true,
    };
  }

  // Namespace page: "<module>/namespaces/<...>.md" (the <...> may itself nest).
  const nsIdx = parts.indexOf('namespaces');
  if (nsIdx === -1) return null;
  const ns = parts.slice(nsIdx + 1).join('/').replace(/\.md$/, '');
  const nsSlug = ns.replace(/\//g, '-');
  return {
    module,
    ns,
    permalink: `/api/${mslug}/${nsSlug}`,
    title: ns,
    fileSlug: `${mslug}-${nsSlug}`,
    isModuleIndex: false,
  };
}

function frontMatter(meta) {
  const lines = ['---', 'layout: default', `title: ${yaml(meta.title)}`];
  if (meta.isModuleIndex) {
    lines.push('parent: API Reference', `nav_order: ${MODULES[meta.module].navOrder}`, 'has_children: true');
  } else {
    lines.push(`parent: ${yaml(MODULES[meta.module].title)}`, 'grand_parent: API Reference');
  }
  lines.push(`permalink: ${meta.permalink}`, '---', '');
  return lines.join('\n');
}

// Quote a YAML scalar only when it contains characters that need it.
function yaml(s) {
  return /[:#]/.test(s) ? JSON.stringify(s) : s;
}

function main() {
  console.log('Running TypeDoc...');
  execSync('npx typedoc', { cwd: ROOT, stdio: 'inherit' });

  const files = walk(RAW);
  // Map every source file -> its destination permalink, for link rewriting.
  const permalinkOf = new Map();
  const pages = [];
  for (const abs of files) {
    const rel = posix.normalize(relative(RAW, abs).split(/[/\\]/).join('/'));
    const meta = classify(rel);
    if (!meta) continue;
    permalinkOf.set(rel, meta.permalink);
    pages.push({ abs, rel, meta });
  }

  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  let written = 0;
  for (const { abs, rel, meta } of pages) {
    let body = readFileSync(abs, 'utf8');
    // TypeDoc may emit its own front matter (from frontmatterGlobals) — strip it.
    body = body.replace(/^---\n[\s\S]*?\n---\n/, '');

    // Rewrite relative *.md links to permalinks (keep any #anchor).
    body = body.replace(/\]\(([^)]+?\.md)(#[^)]*)?\)/g, (m, target, anchor = '') => {
      if (/^https?:/.test(target)) return m;
      const resolved = posix.normalize(posix.join(posix.dirname(rel), target));
      const perma = permalinkOf.get(resolved);
      return perma ? `](${perma}${anchor})` : m;
    });

    writeFileSync(join(OUT, `${meta.fileSlug}.md`), frontMatter(meta) + body);
    written++;
  }

  rmSync(RAW, { recursive: true, force: true });
  console.log(`\nWrote ${written} API pages to docs/api/`);
}

main();
