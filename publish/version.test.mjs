import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { computeVersion } from './version.mjs';

let cwd;

function git(...args) {
  return execFileSync('git', ['-c', 'user.name=test', '-c', 'user.email=test@example.com', ...args], {
    cwd,
    encoding: 'utf8',
  }).trim();
}

function commit() {
  git('commit', '--allow-empty', '--quiet', '--message', 'commit');
  return git('rev-parse', '--short=7', 'HEAD');
}

function version(ref, at = 'HEAD') {
  return computeVersion({ ref, commit: at, cwd });
}

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
  commit();
  assert.equal(version('release/v0.5').version, '0.5.1');
  git('tag', '--annotate', 'v0.5.1', '--message', 'Release 0.5.1');
  assert.equal(version('release/v0.5').version, '0.5.1');
  commit();
  assert.equal(version('release/v0.5').version, '0.5.2');
  git('checkout', '--quiet', 'main');
});

test('a release branch ignores newer minor tags on the same commit', () => {
  git('tag', 'v0.6', 'release/v0.5');
  assert.equal(version('release/v0.5', 'release/v0.5').version, '0.5.2');
  git('tag', '--delete', 'v0.6');
});

test('a release branch whose name disagrees with its tag is refused', () => {
  assert.throws(() => version('release/v0.4', 'release/v0.5'));
});
