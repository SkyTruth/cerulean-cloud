---
name: cerulean-external-dependency-packaging
description: Verify Cerulean external Python and service dependencies before adding or changing pins, especially private GitHub repositories, GitHub archive URLs, subdirectory installs, Docker builds without git, CI install paths, and cross-repo SkyTruth SDK adoption.
---

# Cerulean External Dependency Packaging

Use this skill when adding or changing a dependency in Cerulean that is not a normal public PyPI version pin.

## Use when

- Adding a GitHub URL dependency to any `requirements.txt`.
- Pinning a dependency to a commit SHA, branch, tag, archive URL, or subdirectory install.
- Consuming another SkyTruth repo as an SDK or package.
- Changing service-image dependencies used by Cloud Run, Cloud Functions, Pulumi image builds, or GitHub Actions.
- A dependency works locally but fails in CI, Docker, or deploy.

## Do not use when

- The change is a normal public PyPI package version bump.
- The task is only application logic with no dependency install change.
- The dependency is already installed and no install path or pin is changing.

## Core rules

1. Treat the runtime installer as the source of truth.
- Inspect the relevant GitHub Actions workflow.
- Inspect the relevant Dockerfile or build config.
- Check whether the installer has `git`, SSH credentials, GitHub tokens, BuildKit secrets, or only plain HTTPS.

2. Verify the distribution URL from the same access model the runtime will use.
- For unauthenticated archive URLs, use `curl -I -L`.
- For private GitHub repos, check visibility with `gh repo view`.
- Do not assume local SSH access proves CI or Docker access.
- Do not assume a local editable install proves a requirement file is installable.

3. Private repos require an explicit packaging plan.
Choose one intentionally:
- publish the SDK to an accessible package registry
- make the repo public
- configure CI and Docker build auth for a private repo
- vendor the package as a last resort

4. Keep Docker constraints explicit.
- If the Dockerfile does not install `git`, avoid `git+...` requirements unless the Dockerfile is intentionally updated.
- If auth is needed during Docker build, use a deliberate secret mechanism; do not bake credentials into images or requirement files.
- Prefer immutable, reachable artifacts for production builds.

5. Keep GitHub Actions constraints explicit.
- The default `GITHUB_TOKEN` usually does not grant read access to sibling private repositories.
- If a cross-repo token or GitHub App token is required, name the secret and document the access scope.
- Do not add a dependency pin that requires an undeclared secret.

6. Validate the install path, not just imports.
- Run the targeted tests after dependency changes.
- Also run a dependency install smoke check that matches the failing surface when practical.
- If validating a service requirement, prefer the service's requirement file and installer flags.

## Negative examples

- Pinning `https://github.com/org/private-repo/archive/<sha>.zip` because it works in a local checkout.
- Switching to `git+ssh://git@github.com/...` without wiring SSH credentials into GitHub Actions and Docker builds.
- Running tests after `pip install -e ../other-repo` and treating that as proof the committed requirement works.
- Assuming a GitHub Actions `GITHUB_TOKEN` can read another private repository.
- Adding a Docker build dependency that requires `git` when the service Dockerfile does not install `git`.

## Workflow

1. Identify every install surface touched by the dependency.
- Root test install.
- Service-specific requirement install.
- Docker image build.
- Pulumi image build.
- Cloud Function build, if applicable.

2. Inspect the installer.
- Find the exact `pip install` command.
- Note whether it is isolated, target-based, cached, or run inside Docker.
- Note whether credentials are available.

3. Verify reachability.
- Check repo/package visibility.
- Check archive/raw URL access if using HTTPS.
- Check whether the exact commit/tag is reachable from the runtime access path.

4. Choose the smallest durable packaging fix.
- Public package or public immutable archive if available.
- Authenticated install only when the repo must remain private.
- Docker/CI secret plumbing only when intentionally accepted.
- Vendoring only when no better packaging path exists.

5. Validate and report.
- Run `git diff --check`.
- Run targeted tests through `cerulean-cloud-test-env` when tests are affected.
- Report install-surface validation separately from unit-test validation.

## Output checklist

Before finalizing, state:

- repo/package visibility checked
- install URL or package source is reachable from the intended runtime
- Dockerfile constraints checked
- GitHub Actions constraints checked
- local editable installs did not substitute for committed install validation
- targeted tests passed or the exact blocker is stated
