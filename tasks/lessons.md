# Lessons

- Fetch remote refs and compare app branches before reviewing the latest GitHub code; the repository's default branch can lag main. State the exact reviewed commit.
- When restricting private training endpoints, preserve authenticated assigned-agent data transfers, including sandboxed subprocesses.
- Test helpers must isolate every default path (status, state, downloads, models), not only when a test opts in;
  check ~/.slashcompute for files changed by a test run.
- Cached renders (renderOnce) must key on everything their markup reads, including empty-state text.
- On macOS, Python 3.13 skips .pth files with the UF_HIDDEN flag; a venv under a dot-directory (.claude/worktrees) can get it, so the editable install silently vanishes for subprocesses ("No module named slashcompute"). Fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.
- Parallel fixer agents: worktrees start from the default branch, not the integration branch; tell them the exact tip to reset to.
- Never resolve source conflicts by keeping both sides; a "union" is only safe when each side adds whole new top-level functions. Merge tests by function blocks and check imports.
- Many agents running test suites at once overload the machine (load ~127 on 10 cores) and make timing tests flaky; have fixers run only related tests and run the full suite once at the end.
