#!/usr/bin/env node
// Cut a release of @dbos-inc/vercel-ai in one command.
//
//   node publish/make_release.mjs --version 0.6    tag main as v0.6, push it with a new release/v0.6 branch, publish
//   node publish/make_release.mjs --patch 0.6      tag the tip of release/v0.6 as v0.6.Z, push it, publish
//   ... --no-publish                               tag and push only; skip running the publish workflow
//
// Everything is checked and the tag verified locally before anything is pushed, and each push is atomic, so a
// refusal leaves origin untouched. Every step is safe to rerun: a version whose tag is already on origin but
// is missing from npm has only its publish repeated. Versions come from publish/version.mjs.

import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { computeVersion } from './version.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WORKFLOW = 'publish_npm.yml';
const VERSION = /^(\d+)\.(\d+)$/;

class ReleaseError extends Error {}

function git(args, options = {}) {
  return (
    execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...options }) ?? ''
  ).trim();
}

function gitSucceeds(args) {
  return spawnSync('git', args, { cwd: ROOT, stdio: 'ignore' }).status === 0;
}

function gh(args, options = {}) {
  return (
    execFileSync('gh', args, { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], ...options }) ?? ''
  ).trim();
}

function main() {
  const { values } = parseArgs({
    options: {
      version: { type: 'string' },
      patch: { type: 'string' },
      'no-publish': { type: 'boolean', default: false },
    },
  });
  if ((values.version === undefined) === (values.patch === undefined)) {
    throw new ReleaseError('Pass exactly one of --version X.Y or --patch X.Y');
  }
  const version = parseVersion(values.version ?? values.patch);
  const branch = `release/v${version}`;

  gh(['auth', 'status']);
  git(['fetch', '--tags', 'origin']);

  const step = values.patch !== undefined ? planPatch(branch) : planMinor(version, branch);
  if (step.tag) {
    createTag(step.tag, step.version, step.commit, branch);
    try {
      git(['push', '--atomic', 'origin', ...step.refs], { stdio: 'inherit' });
    } catch {
      // The atomic push left origin untouched, so deleting the local tag lets a rerun recreate it.
      git(['tag', '--delete', step.tag]);
      throw new ReleaseError('Push failed; fix the cause and rerun the same command');
    }
    console.log(`Pushed ${step.refs.map((ref) => ref.replace(/^.*refs\/(tags|heads)\//, '')).join(' and ')}`);
  }
  console.log(`Publishing ${step.version} from ${branch}`);
  if (!values['no-publish']) publish(branch);
}

// Tag main as vX.Y and push it with a new release/vX.Y branch, unless origin already has that tag at main.
function planMinor(version, branch) {
  const tag = `v${version}`;
  checkReady();
  const head = git(['rev-parse', 'HEAD']);
  const released = remoteTagCommit(tag);
  if (released !== undefined) {
    if (released !== head) {
      throw new ReleaseError(`${tag} already exists on origin at ${released.slice(0, 7)}, not at main`);
    }
    if (!gitSucceeds(['rev-parse', '--verify', '--quiet', `origin/${branch}`])) {
      throw new ReleaseError(`${tag} is on origin but ${branch} is not; push the branch by hand`);
    }
    if (isPublished(`${version}.0`)) throw new ReleaseError(`${version}.0 is already released; nothing to do`);
    console.log(`${tag} is on origin but ${version}.0 is not on npm; publishing it again`);
    return { version: `${version}.0` };
  }
  const existing = exactTag('HEAD');
  if (existing) throw new ReleaseError(`main has no commits since ${existing}; nothing to release`);
  // A lower tag on main would become the nearest one and drag preview versions backwards.
  const previous = latestTag();
  if (previous && compareVersions(version, previous.replace(/^v/, '')) <= 0) {
    throw new ReleaseError(`Version ${version} is not above the latest release ${previous}`);
  }
  checkRefAbsent(`refs/tags/${tag}`);
  checkRefAbsent(`refs/heads/${branch}`);
  checkUnpublished(`${version}.0`);
  return { version: `${version}.0`, tag, commit: 'HEAD', refs: [`refs/tags/${tag}`, `HEAD:refs/heads/${branch}`] };
}

// Tag the tip of origin/release/vX.Y as vX.Y.Z, unless it is already tagged.
function planPatch(branch) {
  const tip = `origin/${branch}`;
  if (!gitSucceeds(['rev-parse', '--verify', '--quiet', tip])) {
    throw new ReleaseError(`${branch} does not exist on origin`);
  }
  // A dispatched workflow runs the file as committed on that branch, so the branch must carry this scheme.
  if (!gitSucceeds(['cat-file', '-e', `${tip}:publish/version.mjs`])) {
    throw new ReleaseError(`${branch} predates tag-based versioning and cannot be patched with this script`);
  }
  const { version } = computeVersion({ ref: branch, commit: tip, cwd: ROOT });
  const tagged = exactTag(tip);
  if (tagged && isPublished(version)) {
    throw new ReleaseError(`${branch} has no commits since ${tagged}; nothing to do`);
  }
  if (tagged) {
    // A previous run tagged the tip but its publish did not complete; only the publish is repeated.
    console.log(`${branch} is tagged ${tagged} but ${version} is not on npm; publishing it again`);
    return { version };
  }
  const tag = `v${version}`;
  checkRefAbsent(`refs/tags/${tag}`);
  checkUnpublished(version);
  return { version, tag, commit: tip, refs: [`refs/tags/${tag}`] };
}

function parseVersion(input) {
  const match = VERSION.exec(input);
  if (!match) throw new ReleaseError(`Invalid version "${input}"; expected X.Y`);
  return `${Number(match[1])}.${Number(match[2])}`;
}

function compareVersions(a, b) {
  const [aMajor, aMinor] = a.split('.').map(Number);
  const [bMajor, bMinor] = b.split('.').map(Number);
  return aMajor - bMajor || aMinor - bMinor;
}

function checkReady() {
  if (git(['status', '--porcelain']) !== '') {
    throw new ReleaseError('Working tree is not clean');
  }
  if (git(['rev-parse', '--abbrev-ref', 'HEAD']) !== 'main') {
    throw new ReleaseError('Releases are cut from main; check out main first');
  }
  if (git(['rev-parse', 'HEAD']) !== git(['rev-parse', 'origin/main'])) {
    throw new ReleaseError('Local main differs from origin/main');
  }
}

function checkRefAbsent(ref) {
  if (gitSucceeds(['rev-parse', '--verify', '--quiet', ref])) {
    throw new ReleaseError(`${ref} already exists locally`);
  }
  if (gitSucceeds(['ls-remote', '--exit-code', 'origin', ref])) {
    throw new ReleaseError(`${ref} already exists on origin`);
  }
}

// The commit a tag points at on origin, or undefined if origin has no such tag.
function remoteTagCommit(tag) {
  const out = git(['ls-remote', 'origin', `refs/tags/${tag}`, `refs/tags/${tag}^{}`]);
  if (out === '') return undefined;
  // The peeled entry names the commit of an annotated tag; a lightweight tag has only the plain entry.
  const lines = out.split('\n').map((line) => line.split('\t'));
  return (lines.find(([, ref]) => ref.endsWith('^{}')) ?? lines[0])[0];
}

// The v* tag sitting exactly on a commit, or undefined.
function exactTag(commit) {
  const result = spawnSync('git', ['describe', '--tags', '--exact-match', '--match', 'v[0-9]*', commit], {
    cwd: ROOT,
    encoding: 'utf8',
  });
  return result.status === 0 ? result.stdout.trim() : undefined;
}

// The newest v* tag reachable from HEAD, or undefined.
function latestTag() {
  const result = spawnSync('git', ['describe', '--tags', '--abbrev=0', '--match', 'v[0-9]*', 'HEAD'], {
    cwd: ROOT,
    encoding: 'utf8',
  });
  return result.status === 0 ? result.stdout.trim() : undefined;
}

// Tag a commit locally and confirm a build of the branch there would be versioned exactly as the tag says.
function createTag(tag, version, commit, branch) {
  git(['tag', '--annotate', tag, '--message', `Release ${version}`, commit]);
  const computed = computeVersion({ ref: branch, commit, cwd: ROOT }).version;
  if (computed !== version) {
    git(['tag', '--delete', tag]);
    throw new ReleaseError(`A build of ${branch} would be versioned ${computed}, not ${version}`);
  }
}

// The patch numbers the package has on npm within a minor.
function publishedPatches(major, minor) {
  const name = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).name;
  const result = spawnSync('npm', ['view', name, 'versions', '--json'], { encoding: 'utf8' });
  // A package not yet on npm has nothing published.
  if (result.status !== 0 && result.stderr.includes('E404')) return [];
  if (result.status !== 0) throw new ReleaseError(`npm view ${name} failed:\n${result.stderr}`);
  const parsed = JSON.parse(result.stdout);
  const pattern = new RegExp(`^${major}\\.${minor}\\.(\\d+)$`);
  return (Array.isArray(parsed) ? parsed : [parsed])
    .map((v) => pattern.exec(v))
    .filter(Boolean)
    .map((m) => Number(m[1]));
}

function isPublished(version) {
  const [major, minor, patch] = version.split('.').map(Number);
  return publishedPatches(major, minor).includes(patch);
}

// Refuse a version that would republish or fall below what the package already has on npm.
function checkUnpublished(version) {
  const [major, minor, patch] = version.split('.').map(Number);
  const published = publishedPatches(major, minor);
  if (published.length > 0 && Math.max(...published) >= patch) {
    throw new ReleaseError(
      `Version ${version} is not above ${major}.${minor}.${Math.max(...published)}, which is already on npm`,
    );
  }
}

// Run the publish workflow on a branch and wait for it to finish.
function publish(branch) {
  const before = latestRun(branch);
  gh(['workflow', 'run', WORKFLOW, '--ref', branch]);
  let run;
  for (let attempt = 0; attempt < 60 && (run === undefined || run === before); attempt++) {
    sleep(2000);
    run = latestRun(branch);
  }
  if (run === undefined || run === before) {
    throw new ReleaseError(`Publish run for ${branch} did not start; check the Actions tab`);
  }
  console.log(`Publishing from ${branch}: ${gh(['run', 'view', String(run), '--json', 'url', '--jq', '.url'])}`);
  gh(['run', 'watch', String(run), '--exit-status'], { stdio: 'inherit' });
  console.log(`Published ${branch}`);
}

function latestRun(branch) {
  const runs = JSON.parse(
    gh(['run', 'list', '--workflow', WORKFLOW, '--branch', branch, '--limit', '1', '--json', 'databaseId']),
  );
  return runs.length > 0 ? runs[0].databaseId : undefined;
}

function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

try {
  main();
} catch (error) {
  // Failed git and gh commands have already written their own stderr.
  console.error(`error: ${error.message}`);
  process.exit(1);
}
