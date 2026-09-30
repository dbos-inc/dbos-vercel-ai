import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { computeVersion } from './version.mjs';

let cwd;
let commits = 0;

function git(...args) {
  const config = ['user.name=test', 'user.email=test@example.com', 'commit.gpgSign=false', 'tag.gpgSign=false'];
  return execFileSync('git', [...config.flatMap((c) => ['-c', c]), ...args], { cwd, encoding: 'utf8' }).trim();
}

// Unique messages, so commits on different branches never collapse into one hash.
function commit() {
  git('commit', '--allow-empty', '--quiet', '--message', `commit ${++commits}`);
  return git('rev-parse', 'HEAD');
}

function version(ref, at = 'HEAD') {
  return computeVersion({ ref, commit: at, cwd });
}

// main:         v0.5 - m1 - m2
// release/v0.5: v0.5 - r1 (v0.5.1) - r2
before(() => {
  cwd = mkdtempSync(join(tmpdir(), 'version-test-'));
  git('init', '--quiet', '--initial-branch', 'main');
  commit();
  git('tag', 'v0.5'); // lightweight, as nbgv left the existing tags
  git('branch', 'release/v0.5');
  commit();
  commit();
});

after(() => rmSync(cwd, { recursive: true, force: true }));

test('main builds a preview of the next minor', () => {
  assert.deepEqual(version('main'), { version: '0.6.2-preview', distTag: 'preview' });
});

test('other branches build a test version', () => {
  const sha = git('rev-parse', '--short=7', 'HEAD');
  assert.deepEqual(version('feature'), { version: `0.6.2-test.${sha}`, distTag: 'test' });
});

test('release branches count patches from their tags', () => {
  assert.deepEqual(version('release/v0.5', 'release/v0.5'), { version: '0.5.0', distTag: 'latest' });
  git('checkout', '--quiet', 'release/v0.5');
  const r1 = commit();
  assert.notEqual(r1, git('rev-parse', 'main~1'), 'release branch must diverge from main');
  assert.equal(version('release/v0.5').version, '0.5.1');
  git('tag', '--annotate', 'v0.5.1', '--message', 'Release 0.5.1');
  assert.equal(version('release/v0.5').version, '0.5.1');
  commit();
  assert.equal(version('release/v0.5').version, '0.5.2');
  git('checkout', '--quiet', 'main');
  assert.equal(version('main').version, '0.6.2-preview', 'patches must not affect main');
});

test('a release branch ignores newer minor tags on its tagged commits', () => {
  // v0.6 shares the v0.5.1 commit, so only the release-branch filter keeps the build on the 0.5 line.
  git('tag', 'v0.6', 'release/v0.5~1');
  try {
    assert.equal(version('release/v0.5', 'release/v0.5').version, '0.5.2');
  } finally {
    git('tag', '--delete', 'v0.6');
  }
});

test('a patch-floor tag above the height sets the patch', () => {
  // Migration case: nbgv published 0.4.4 from the commit tagged v0.4.
  git('checkout', '--quiet', '--orphan', 'release/v0.4');
  const base = commit();
  git('tag', 'v0.4');
  git('tag', '--annotate', 'v0.4.4', '--message', 'Floor at 0.4.4');
  assert.equal(version('release/v0.4').version, '0.4.4');
  commit();
  assert.equal(version('release/v0.4').version, '0.4.5');
  assert.equal(version('release/v0.4', base).version, '0.4.4');
  git('checkout', '--quiet', 'main');
});

test('a new minor on main restarts preview numbering without moving older lines', () => {
  git('checkout', '--quiet', '-b', 'scratch-main', 'main');
  git('tag', '--annotate', 'v0.6', '--message', 'Release 0.6');
  assert.equal(version('release/v0.6').version, '0.6.0');
  commit();
  assert.equal(version('main').version, '0.7.1-preview');
  assert.equal(version('release/v0.5', 'release/v0.5').version, '0.5.2');
  git('checkout', '--quiet', 'main');
  git('tag', '--delete', 'v0.6');
});

test('a release branch with no tag for its minor is refused', () => {
  assert.throws(() => version('release/v0.3', 'release/v0.5'), /No names found/);
});
