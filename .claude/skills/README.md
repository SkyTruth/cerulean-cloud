# Cerulean Skill Catalog

This directory is the canonical checked-in skill catalog for Cerulean Cloud.

- Claude discovers project skills here through `.claude/skills`.
- Codex discovers the same skills through `.agents/skills`, which is a symlink to this directory.
- Keep one stored copy of each skill here.
- Keep Cerulean-specific skills under `cerulean-*` names and SkyTruth-wide repo policy skills under `skytruth-*` names.
- Keep shared `SKILL.md` files portable: `name`, `description`, concise Markdown instructions, and optional `references/`, `scripts/`, or `assets/`.
- Put Codex-specific metadata in `agents/openai.yaml`.
