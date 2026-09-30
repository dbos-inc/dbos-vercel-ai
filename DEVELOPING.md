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

Release from the tip of `main` with one command, giving the version explicitly:

```sh
npm run release -- --version 0.6
```

With a clean `main` identical to `origin/main`, this tags `main` as `v0.6`,
pushes the tag together with a new `release/v0.6` branch, then runs the publish
workflow on that branch and waits for it. Everything is checked before anything
is pushed, and no commit is made on any branch.

To patch a release, merge the fix into `release/vX.Y`, then:

```sh
npm run release -- --patch 0.6
```

This tags the tip of the branch `v0.6.Z`, pushes the tag, and publishes from the
branch. Every published release is a tag; never add a release tag by hand.

Both commands are safe to rerun: a tag already on origin whose version never
reached npm has only its publish repeated. Pass `--no-publish` to tag and push
without publishing. You need the [GitHub CLI](https://cli.github.com/) logged
in with the `repo` and `workflow` scopes, and permission to push tags and
branches. Release branches from before tag-based versioning (`release/v0.5` and
earlier) cannot be patched this way.

## Versions

The committed `package.json` version is the placeholder `0.0.0-placeholder`.
`publish/version.mjs` derives the real version from the nearest `vX.Y[.Z]` tag
and the commits since it, and the publish workflow stamps it at publish time.
Run it on any checkout to see what a build of the current branch would be:

| Branch         | Version                                         | npm dist-tag |
| -------------- | ----------------------------------------------- | ------------ |
| `release/vX.Y` | `X.Y.0` at tag `vX.Y`, then `X.Y.Z` at `vX.Y.Z` | `latest`     |
| `main`         | `X.(Y+1).<commits since tag>-preview`           | `preview`    |
| anything else  | `X.(Y+1).<commits since tag>-test.<sha>`        | `test`       |

## Publishing

`.github/workflows/publish_npm.yml` runs on every push to `main`, so each merge
publishes a preview. Dispatching it manually on any other branch publishes a
`test` build, which exercises the publish path without touching `latest` or
`preview`. It authenticates to npm with
[trusted publishing](https://docs.npmjs.com/trusted-publishers) (OIDC), so no
npm token is stored in the repo.
