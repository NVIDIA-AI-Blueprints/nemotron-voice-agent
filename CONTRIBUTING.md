# Contributing Guidelines

Use the following guidelines to contribute to this project.

## Pull Requests

Use the following workflow for code and documentation contributions:

1. Create a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
   of this repository.
2. Clone your fork and create a branch for the change.
3. Run the applicable formatting, linting, test, client-build, and documentation
   checks locally.
4. Complete the pull request template, including one **Type of Change** choice
   and the documentation writer review receipt.
5. Push the branch to your fork and open a pull request against the appropriate
   upstream branch.
6. If you are contributing for the first time, download the
   [Contribution License Agreement](CLA.md) and share a signed copy.

## Documentation Writer Review Receipt

Pull requests that change code or documentation must record a documentation
review after the changes and applicable validation are complete. In the pull
request description:

1. Check **Documentation writer reviewed the completed changes**.
2. Keep one result: `docs-updated`, `no-docs-needed`, or `blocked`.
3. Add the changed documentation paths or a concise rationale to **Evidence**.
4. Record the agent product and surface that performed the review.
5. After committing the reviewed changes, populate the hidden metadata with:

   ```bash
   git rev-parse --short HEAD
   git rev-parse --short HEAD:AGENTS.md
   ```

Rerun the review and refresh both values after any later commit. The
`CI / Documentation Writer Review` workflow reports missing, invalid, or stale
receipts in advisory mode.

Maintainers can measure adoption with the following command. Replace the date
with the start of the reporting window:

```bash
python scripts/docs-review-receipt.py report --since 2026-08-01 --format summary
```

The report uses the authenticated GitHub CLI session and also supports `json`
and `csv` output.
