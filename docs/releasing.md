# Preparing a release

The distribution name is `ling-aspec`; the project name, Python import and CLI remain ASpec / `aspec`.
Setuptools normalizes archive filenames to `ling_aspec-<version>...`. Do not upload under the
unrelated PyPI project `aspec`.

Wheels contain the runtime, public CLI, model JSON assets, formal recipes, Quickstart conversations,
runtime dependency manifests and the pinned SGLang patch. Source distributions also contain full
documentation, tests, setup and maintenance tools. Canonical resource files live in their existing
source locations; `setup.py` collects them into the wheel during the build.

## Versioning

Current public PyPI releases use the `b0` beta suffix, starting with `0.1.0b0`.
Developers maintain the major/minor/patch version in `pyproject.toml` and retain that suffix,
for example `0.1.0b0`, `0.2.0b0`, then `0.2.1b0`. Each upload containing changed source or
artifacts needs a new version selected by the developer. The current policy does not advance
the beta suffix to `b1`/`b2` or automatically promote a release to an unsuffixed version.

The release tag is exactly `v` plus that complete version, such as `v0.1.0b0`.
CI reads and validates the selected version; it does not choose or increment it.
The team's current "debug" release designation uses the standard
[Python beta version syntax](https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases).
An exact install pin such as `ling-aspec==0.1.0b0` selects that prerelease explicitly.

## Prepare

1. Select a clean, reviewed commit and set its version in `pyproject.toml`.
2. Update the changelog, user guides, supported paths and relevant examples.
3. Check source attribution, license texts and security-reporting instructions. Model and dataset
   redistribution terms are separate from the source license.
4. Before publishing a new repository, check both files and Git history for private material.
   A reviewed source export need not include development history.

## Verify

Use a dedicated Linux venv and run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md).
Build into an empty output directory:

```bash
python -m tools.build_release --outdir /absolute/path/to/new-release-dist
```

This command builds the sdist and then its wheel, audits both, collects the two runtime requirement
files under `runtime/`, and writes `SHA256SUMS`. It rejects a nonempty output directory.
`python -m build` and `python -m tools.audit_release DIST` remain available separately.

[GitHub CI](../.github/workflows/ci.yml) runs source quality checks and CPU/offline contracts on pull
requests and configured branch pushes. It does not build distributions. Packaging-related source
contracts remain in that suite so resource and audit regressions can be found before tagging.

The separate [release build workflow](../.github/workflows/build-release.yml) supports manual
validation and runs when a `v*` tag is pushed to GitHub. A default manual run builds and checks
artifacts without publishing. Creating a local tag alone does not trigger it. On tag pushes, the tag must exactly match `v` plus
the version in `pyproject.toml`, for example `v0.1.0b0` for `0.1.0b0`; a mismatch fails before
building. The workflow runs the same build command, checks distribution metadata with
`twine check --strict`, saves the audited release bundle as the
`aspec-distributions` workflow artifact, and runs a dependent `wheel-install` job on Python 3.11.
That job checks SHA256s, installs the built wheel in a dedicated CPU venv, and executes:

```bash
python tools/check_wheel_install.py
```

The check changes to a temporary working directory and verifies installed module origins, CLI help,
recipe copying and Plan generation, both Quickstart Plans, read-only doctor, package/build identities,
and actual SGLang patch application, repetition and conflict rejection. It uses the CPU CI dependency
subset and the official SGLang wheel without its GPU dependencies. Missing full-runtime requirements
must keep doctor non-ready; this fixture is not production dependency or GPU/training acceptance.

- Include runtime/model assets, patches, requirements, guides, recipes, source tools, source
  headers and applicable license texts.
- Exclude credentials, private endpoints, local environments, datasets, weights, checkpoints,
  profiler traces and generated logs. Build from the reviewed tree, not a dirty working directory.
- Unpack the sdist into a fresh directory and check documentation and tool entry points there.
- Install the wheel in a fresh venv and run `aspec patch-sglang` from outside the source tree
  against the pinned SGLang installation. Verify application, repeat application and read-only
  `--check`; wrong versions or conflicting hunks must fail without overwriting files.
- In a fresh environment, follow installation and Quickstart through training, recovery and export.
  For runtime changes, run the relevant [GPU, numerical and fault checks](contributing/validation.md).
  A documentation-only change does not require a new all-algorithm training matrix.

Benchmark pages should contain results, methods, relevant limitations and reproduction entry points.
Historical raw records and per-artifact hash inventories are not required release assets.
Evidence records the effective installed package hash (including the bundled patch and assets),
distribution version and build provenance when available. A wheel does not borrow Git identity from
an enclosing checkout; unavailable build revision remains unknown.

## Rehearse before publishing

### Build and install without uploading

First run the source CI for the same commit and require its quality and CPU contract jobs to pass.
The release workflow checks packaging and wheel installation; it does not rerun that source suite.

Once this workflow configuration is on the repository's default branch, open **Actions → Build and
publish release → Run workflow**, select the branch to check, and run the workflow.
GitHub requires the manual trigger to be present on the default branch before exposing this entry;
see [manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

This runs `build` and `wheel-install`, retaining the audited bundle in `aspec-distributions`.
The publishing job is skipped. The build and installation jobs receive no publishing secrets.
Manual runs never publish to production PyPI, even when the selected ref is a version tag.
Repeat these checks without changing the version or creating a release tag.

The installation check covers installed command origins, version identity, assets, recipe/Quickstart
Plans and SGLang patch behavior. Before the first production release, download the audited bundle
and follow the [wheel installation guide](user-guide/installation.md#install-a-release-wheel) in a
fresh Linux venv on allocated GPUs. Complete Quickstart training, recovery and export; use the
relevant [validation gates](contributing/validation.md) for runtime changes. The CPU fixture does not
establish GPU or complete production dependency acceptance.

## Publish

### One-time PyPI and GitHub setup

The workflow uses your PyPI API token. Configure it in the repository that publishes releases,
[`inclusionAI/ASpec`](https://github.com/inclusionAI/ASpec), rather than the development fork:

1. Sign in to [PyPI account settings](https://pypi.org/manage/account/) and create or verify an API
   token. The first upload creates `ling-aspec`, so it needs an account-wide token when the project
   does not exist yet. A token scoped to another project cannot create it. After the first successful
   release, replace it with a token scoped to `ling-aspec` and revoke the bootstrap token.
2. Open **Settings → Environments → New environment** in the GitHub repository and name it **`pypi`**.
   Under **Environment secrets**, add **`PYPI_API_TOKEN`** with the entire token value, including
   its `pypi-` prefix. Enter the token directly in GitHub; keep it out of source files and logs.
3. Configure **Required reviewers** on the environment to require approval before each upload.
   Choose maintainers who can approve releases. If the tag creator must approve their own release,
   leave **Prevent self-review** disabled. Environment approval happens after the build/install jobs.
4. Under **Deployment branches and tags**, select **Selected branches and tags** and add a rule
   with ref type **Tag** and pattern **`v*`**. A branch-only rule will not permit version tags.
5. Ensure GitHub Actions is enabled and any organization/repository action allowlist permits the
   actions referenced by the workflow, including `pypa/gh-action-pypi-publish`.

See PyPA's [first-upload instructions](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#create-an-account),
GitHub's [environment configuration guide](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
and the PyPA action's [API token usage](https://github.com/pypa/gh-action-pypi-publish#specifying-a-different-username).
This workflow explicitly uses token authentication and disables attestations, which require Trusted
Publishing. No PyPI Trusted Publisher or GitHub `id-token: write` permission is needed for this setup.

### Release a version

Before pushing a version tag, verify that its exact commit has passed the CI source checks and the
applicable release checks above. The tag workflow does not rerun the source test suite. After pushing,
require both `build` and `wheel-install` to pass; an uploaded artifact alone is not installation
acceptance. A failed run can be rerun for the same tag after an infrastructure issue is resolved;
source fixes require a new reviewed commit and version/tag.

1. Merge the reviewed release changes, including the version in `pyproject.toml`, into the main
   repository. Choose a version that has not already been uploaded or tagged; `0.1.0b0` requires
   exactly `v0.1.0b0`. Creating a GitHub Release for an existing tag does not send a new tag-push event.
2. Push that tag to `inclusionAI/ASpec` at the reviewed commit. Alternatively, in GitHub's
   **Releases → Draft a new release**, create the new tag at that commit and publish the release.
   Mark a GitHub Release for a `b0` version as a pre-release.
   Pushing a version tag authorizes the publishing workflow; with required reviewers configured,
   uploading still waits for environment approval.
3. Open **Actions → Build and publish release**. After `build` and `wheel-install` pass, review the
   `aspec-distributions` artifact and approve the pending `pypi` deployment.
4. The `publish` job downloads that same artifact, verifies `SHA256SUMS`, and uploads only its wheel
   and sdist. It does not rebuild them or upload `runtime/` and `SHA256SUMS` as Python packages.
5. Verify the files and version on [PyPI](https://pypi.org/project/ling-aspec/), then follow the
   [pinned runtime installation guide](user-guide/installation.md) in a fresh Linux venv.

The publishing job runs only in `inclusionAI/ASpec`; forks still build and check artifacts but skip
publication. Production publishing requires a `v*` tag push; manual runs cannot enable it. The
workflow does not create a GitHub Release itself. The `0.1.0b0`
version is a prerelease; installing that version requires an exact pin or `--pre` until a stable
release is available.

Missing tokens, rejected metadata, failed checks and duplicate uploads fail the workflow. PyPI
does not allow replacing an existing distribution file. Before retrying an upload failure, inspect
PyPI for partially uploaded files; a rerun does not silently skip existing files. Keep failures and
their logs, and use a new version for source changes.

Review the exact artifacts, repository destination, version/tag and release notes. Summarize checks
actually performed and any known limitations. Publish only the reviewed source and artifacts;
retain the corresponding build and validation records separately.
