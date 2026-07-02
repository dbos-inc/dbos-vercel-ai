# Developing

## Setup

```sh
npm install
```

## Building

```sh
npm run build   # compile src/ to dist/ (tsconfig.build.json)
npm run lint    # type-check everything, including tests, with no emit
```

## Testing

Tests exercise the middleware against a real DBOS runtime, so they require a
local Postgres database. Set `DBOS_TEST_DB_URL` to override the default
(`postgresql://postgres@localhost:5432/dbos_vercel_ai_test_dbos_sys`):

```sh
npm test
```

`npm test` builds first (via `pretest`), then runs the suite with
`tsx --test`. CI runs the same steps against Postgres 16 on Node 22 and 24; see
`.github/workflows/ci.yml`.

## Validating the package

Before publishing (and in CI's publish job), verify the package is well-formed —
`publint` for `package.json` correctness and `attw` for type-resolution across
module systems:

```sh
npm run check:package   # build, publint, attw --pack .
npm pack --dry-run      # inspect the exact tarball contents
```

# Releasing

This package versions and publishes exactly like the other `@dbos-inc/*`
packages, using [Nerdbank.GitVersioning](https://github.com/dotnet/Nerdbank.GitVersioning)
(NBGV) to derive the version from git history.

## How versioning works

- The committed `package.json` version is the placeholder `0.0.0-placeholder`.
  The real version is never committed — it is computed and stamped at publish
  time so there are no version-bump commits or merge conflicts.
- `version.json` holds the base version (currently `0.1-preview`). NBGV appends a
  git-height component, producing versions like `0.1.42-preview`.
- The `-preview` suffix marks this integration as experimental. While it is
  present, published builds get the `preview` npm dist-tag, so a plain
  `npm install @dbos-inc/vercel-ai` (which resolves `latest`) will not pick them
  up until a stable release exists.

## Publishing

Publishing is manual, via the **Publish to npm** GitHub Action
(`.github/workflows/publish.yml`, triggered with *Run workflow*). On each run it:

1. Installs, then runs `npm run check:package` (build + `publint` + `attw`) and
   the test suite dependencies — a build that fails validation never publishes.
2. Stamps the NBGV-computed version into `package.json`.
3. Publishes to npm with a dist-tag chosen from the branch and version:

   | Branch                    | Version         | npm dist-tag |
   | ------------------------- | --------------- | ------------ |
   | `main` / `release/v*`     | `-preview`      | `preview`    |
   | `main` / `release/v*`     | stable          | `latest`     |
   | any other branch          | any             | `test`       |

   Run the workflow with **dry-run** checked to pack and validate without
   publishing.

Requires an npm automation token in the `NPM_PUBLISH_TOKEN` repository secret,
with publish rights to the `@dbos-inc` scope.

## Cutting a stable (non-preview) release

1. Drop the prerelease tag by running `nbgv prepare-release` (creates a
   `release/v0.1` branch and bumps `main` to the next version), **or** edit
   `version.json` to remove `-preview`.
2. Run the **Publish to npm** workflow from that branch. With no prerelease
   suffix, the build publishes under the `latest` tag.
