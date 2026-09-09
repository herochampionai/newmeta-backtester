"""Push to GitHub helper.
The GitHub token used by opencode doesn't have repo-create scope, so
the user must create the empty repo first at:
   https://github.com/new   (name = newmeta-backtester, public)

Then run this script.

Alternative one-liner after creating the empty repo at github.com/new:
   git push -u origin master
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
REMOTE = "https://github.com/youhannamitri/newmeta-backtester.git"


def main():
    # Check we're in a git repo
    res = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=REPO_DIR,
                          capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] Not a git repo. Run `git init` first.")
        sys.exit(1)

    # Check remote
    res = subprocess.run(["git", "remote", "-v"], cwd=REPO_DIR,
                          capture_output=True, text=True)
    if REMOTE not in res.stdout:
        print(f"Adding remote {REMOTE}")
        subprocess.run(["git", "remote", "add", "origin", REMOTE], cwd=REPO_DIR)

    # Show current status
    print("\nCurrent git status:")
    subprocess.run(["git", "status", "--short"], cwd=REPO_DIR)

    # Push
    print(f"\nPushing to {REMOTE} (branch: master)...")
    res = subprocess.run(["git", "push", "-u", "origin", "master"], cwd=REPO_DIR)
    if res.returncode == 0:
        print("\n[OK] Pushed successfully!")
        print(f"View at: {REMOTE.replace('.git', '')}")
    else:
        print("\n[INFO] Push failed. This is expected if the repo doesn't exist yet.")
        print("Steps:")
        print("  1. Open https://github.com/new")
        print("  2. Repository name: newmeta-backtester")
        print("  3. Public, no README/.gitignore/license (we have them)")
        print("  4. Click 'Create repository'")
        print("  5. Run this script again, OR:")
        print(f"     git push -u {REMOTE} master")


if __name__ == "__main__":
    main()