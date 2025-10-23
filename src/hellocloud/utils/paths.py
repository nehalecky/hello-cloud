"""Path utilities for finding project and repository roots."""

from pathlib import Path


def get_repo_root() -> Path:
    """Find git repository root by looking for .git (file or directory).

    This works in both regular git repositories and git worktrees:
    - Regular repo: .git is a directory
    - Worktree: .git is a file containing a pointer to the git data

    Returns:
        Path to the repository root directory

    Raises:
        FileNotFoundError: If not inside a git repository

    Example:
        >>> from hellocloud.utils import get_repo_root
        >>> data_path = get_repo_root() / "data" / "my_dataset.parquet"
    """
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        git_path = parent / ".git"
        if git_path.exists():  # Works for both file (worktree) and directory
            return parent
    raise FileNotFoundError("Not inside a git repository")
