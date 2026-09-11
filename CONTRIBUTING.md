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
6. Sign off every commit to certify the [Developer Certificate of Origin
   (DCO)](#developer-certificate-of-origin-dco) by using `git commit -s`.

## Developer Certificate of Origin (DCO)

For third-party (external) contributions to the NVIDIA project code, we require a
per-commit sign-off certifying the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO), version 1.1. Add the sign-off automatically with:

```bash
git commit -s
```

This appends a `Signed-off-by: Your Name <your.email@example.com>` line to the
commit message, certifying the text below. The sign-off name and email must match
your real name and a valid email address.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

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
