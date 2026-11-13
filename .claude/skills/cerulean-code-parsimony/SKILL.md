---
name: cerulean-code-parsimony
description: Keep Cerulean backend and orchestrator code changes small, canonical, and contract-driven. Use when implementing or reviewing refactors, config parsing, accessors, validation, edge-case handling, database-client write paths, or tests where the user asks for simplicity, compactness, parsimony, fewer speculative branches, fewer helper layers, or less defensive code.
---

# Cerulean Code Parsimony

Use this skill when implementation shape is flexible and the main risk is overbuilding.

## Do Not Use When

- The task is only running tests. Use `cerulean-cloud-test-env`.
- The task is primarily schema or migration follow-through. Use `cerulean-database-change-checklist`.
- The task is query performance diagnosis. Use `cerulean-db-performance-triage`.
- The task is classification taxonomy boundary work. Use `cerulean-slick-classification-boundary`.

## Core Rules

1. Prefer one canonical config path.
- Do not support aliases unless existing production data requires them.
- If a key is canonical, use that key directly in code and tests.
- Keep internal attribute names aligned with config names when practical.
- Prefer domain names such as `short_name` over generic synonyms such as `key`.

2. Let natural errors work when they are enough.
Before adding `raise`, ask:
- Would the next line fail quickly?
- Would the traceback point to the right code?
- Would the raw error include enough context?
- Would an explicit raise prevent data corruption, unsafe SQL, cache poisoning, silent wrong output, or a misleading domain failure?

3. Keep explicit raises only when they earn their line.
Good reasons include:
- unsafe identifier or injection boundary
- unsupported domain mode where a raw Python error would obscure intent
- persistent bad state such as a corrupt cache file
- silent data loss or wrong result
- ambiguous database state that needs a domain-specific error

4. Put behavior at the owning boundary.
- If logic exists only for one access type, put it on that access type.
- If logic is shared by concrete accessors, put it on the base accessor.
- If logic is only a coordinator concern, keep it in the coordinator.
- Do not keep a free helper solely because it is convenient to test.
- Do not keep both a protocol and a base class unless callers genuinely need both contracts.

5. Split responsibilities before broadening names.
- If a file contains two real systems, prefer splitting it over renaming it to something vague.
- Keep coordinators small; push source-specific loading, SQL, and cache behavior into accessors or focused modules.
- Rename only when the underlying responsibility is already coherent.

6. Do not let test convenience broaden runtime behavior.
- Patch or fake boundaries in tests.
- Do not add local-file fallbacks, alternate config keys, or unused modes solely to make tests easier.

7. Delete before abstracting.
- Remove unreachable compatibility paths first.
- Add helper functions only after duplication is real and current.
- Avoid dataclasses or normalizer layers when a raw row plus one accessor is clearer.

8. Keep tests behavior-facing.
- Assert public payload shape, database side effects, and error contracts that matter.
- Do not test internal helper ceremony unless it is the actual contract.

## Workflow

1. Identify the contract.
- Name the canonical inputs, outputs, and owner of each responsibility.
- Decide which layer owns validation: database constraint, accessor, database client, or caller.
- Trace the actual production path before preserving generic input support.

2. Scan for speculative support.
- Look for aliases, wrappers, custom errors, fallback branches, caches, protocols, and "just in case" paths.
- Remove support for cases that are not reachable or not planned production paths.

3. Apply the raise test.
- Remove explicit raises that only pre-validate natural local failures.
- Keep raises that improve safety, integrity, or domain clarity.

4. Apply the ownership test.
- Ask: would moving this method/function reduce the top-level surface area without hiding behavior?
- Ask: is this generic because multiple real callers need it, or because it was written before the actual path was known?
- Ask: does this file name describe one responsibility, or is it hiding multiple systems?

5. Align tests with the simplified contract.
- Update fixtures to canonical keys.
- Monkeypatch external boundaries instead of adding runtime branches.
- Prefer fewer tests that assert behavior over many tests that assert plumbing.

6. Validate narrowly.
- Run `py_compile`, `git diff --check`, and targeted tests through `cerulean-cloud-test-env`.

## Negative Examples

- Do not accept both `name_col`, `name_field`, and `display_name_field` unless production config already uses all three.
- Do not add a local-file branch to a `GCS` accessor just because tests need a fake file path.
- Do not wrap a missing required dict key in a custom `ValueError` if the next line would raise `KeyError` with the same useful traceback.
- Do not add a shared config dataclass when each accessor can read the few keys it owns.
- Do not keep a module-level DB URL helper when only the remote DB accessor uses it.
- Do not rename a bloated file to a broader name when splitting the responsibilities would make both pieces smaller.

## Output Checklist

Before finalizing:
- one canonical config vocabulary remains
- no unsupported aliases remain
- explicit raises are justified by safety, data integrity, cache integrity, or domain clarity
- test-only convenience does not appear in runtime code
- helper functions are fewer than before unless clearly justified
- ownership boundaries are clear at class and file level
- targeted tests pass or the environment blocker is stated precisely
