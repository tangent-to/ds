#!/usr/bin/env node
/**
 * Version bump script
 *
 * Usage:
 *   node scripts/bump-version.js patch   # 0.3.1 -> 0.3.2
 *   node scripts/bump-version.js minor   # 0.3.1 -> 0.4.0
 *   node scripts/bump-version.js major   # 0.3.1 -> 1.0.0
 *   node scripts/bump-version.js 0.4.0   # Set explicit version
 *
 * Options:
 *   --dry-run    Show what would change without modifying files
 *   --no-git     Skip git commit and tag
 */

import { readFileSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, '..');

// Files that contain version strings
const VERSION_FILES = [
  { path: 'package.json', key: 'version' },
  { path: 'jsr.json', key: 'version' },
  { path: 'deno.json', key: 'version' },
  { path: 'CITATION.cff', pattern: /^version:\s*".*"$/m, replace: (v) => `version: "${v}"` }
];

function readJSON(filepath) {
  return JSON.parse(readFileSync(join(ROOT, filepath), 'utf8'));
}

function writeJSON(filepath, data) {
  writeFileSync(join(ROOT, filepath), JSON.stringify(data, null, 2) + '\n');
}

function readFile(filepath) {
  return readFileSync(join(ROOT, filepath), 'utf8');
}

function writeFile(filepath, content) {
  writeFileSync(join(ROOT, filepath), content);
}

function parseVersion(version) {
  const match = version.match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!match) throw new Error(`Invalid version format: ${version}`);
  return {
    major: parseInt(match[1], 10),
    minor: parseInt(match[2], 10),
    patch: parseInt(match[3], 10)
  };
}

function bumpVersion(current, type) {
  const v = parseVersion(current);

  switch (type) {
    case 'major':
      return `${v.major + 1}.0.0`;
    case 'minor':
      return `${v.major}.${v.minor + 1}.0`;
    case 'patch':
      return `${v.major}.${v.minor}.${v.patch + 1}`;
    default:
      // Assume explicit version
      parseVersion(type); // Validate format
      return type;
  }
}

function updateFile(fileConfig, newVersion, dryRun) {
  const { path, key, pattern, replace } = fileConfig;
  const fullPath = join(ROOT, path);

  try {
    if (key) {
      // JSON file
      const data = readJSON(path);
      const oldVersion = data[key];
      data[key] = newVersion;

      if (!dryRun) {
        writeJSON(path, data);
      }

      return { path, oldVersion, newVersion, success: true };
    } else if (pattern) {
      // Text file with pattern
      let content = readFile(path);
      const match = content.match(pattern);
      const oldVersion = match ? match[0] : 'unknown';
      content = content.replace(pattern, replace(newVersion));

      if (!dryRun) {
        writeFile(path, content);
      }

      return { path, oldVersion, newVersion: replace(newVersion), success: true };
    }
  } catch (err) {
    return { path, error: err.message, success: false };
  }
}

function gitCommitAndTag(version, dryRun) {
  const commands = [
    `git add package.json jsr.json deno.json CITATION.cff`,
    `git commit -m "chore: bump version to ${version}"`,
    `git tag v${version}`
  ];

  console.log('\nGit commands:');
  for (const cmd of commands) {
    console.log(`  ${cmd}`);
    if (!dryRun) {
      try {
        execSync(cmd, { cwd: ROOT, stdio: 'inherit' });
      } catch (err) {
        console.error(`Failed: ${cmd}`);
        process.exit(1);
      }
    }
  }

  console.log(`\nTo publish, run:`);
  console.log(`  git push origin main`);
  console.log(`  git push origin v${version}`);
}

// Main
const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const noGit = args.includes('--no-git');
const bumpType = args.find(a => !a.startsWith('--'));

if (!bumpType) {
  console.log('Usage: node scripts/bump-version.js <patch|minor|major|x.y.z> [--dry-run] [--no-git]');
  process.exit(1);
}

// Get current version
const pkg = readJSON('package.json');
const currentVersion = pkg.version;
const newVersion = bumpVersion(currentVersion, bumpType);

console.log(`\nVersion bump: ${currentVersion} -> ${newVersion}`);
if (dryRun) console.log('(dry run - no files will be modified)\n');

// Update all files
console.log('\nUpdating files:');
const results = VERSION_FILES.map(f => updateFile(f, newVersion, dryRun));

for (const result of results) {
  if (result.success) {
    console.log(`  ✓ ${result.path}`);
  } else {
    console.log(`  ✗ ${result.path}: ${result.error}`);
  }
}

// Git operations
if (!noGit && !dryRun) {
  gitCommitAndTag(newVersion, dryRun);
} else if (!noGit && dryRun) {
  console.log('\nWould run git commands (use --no-git to skip):');
  gitCommitAndTag(newVersion, true);
}

console.log('\nDone!');
