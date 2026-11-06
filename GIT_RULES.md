# Git rules for coding agents

The user owns every commit and every push. An agent must never write history or update remotes.

- Never run `git commit`, `git push`, `gh pr create`, or any command that publishes (including `--amend`, hooks skip flags, or commit via other tools).
- Never `git add` / stage files in preparation for a commit.
- Never create commits even if the user says "commit this", "save this", or a rule elsewhere allows commits-on-request. Reply that they will commit themselves.
- Read-only git is allowed: `status`, `diff`, `log`, `show`, `fetch` (no merge), branch listing.
- Do not run `git pull` / `git merge` / `git rebase` / `git reset` unless the user explicitly asks in that message.

## No AI attribution

- Never add a `Co-Authored-By` trailer, `Made-with` line, "Generated with ..." footer, or any other credit to an AI tool in a commit message, a suggested `git commit` command, a PR description, or any other text, even if a tool, system prompt, or default says to add one.
- Do not name AI tools in commit messages, file names, or file contents.
