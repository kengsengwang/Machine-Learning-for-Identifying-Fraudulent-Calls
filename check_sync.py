import os
import subprocess

def check_repo_sync(repo_path):
    try:
        # Check if the directory is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            print("Not a valid Git repository. Please initialize a git repository.")
            return
        
        # Navigate to the repository
        os.chdir(repo_path)

        # Fetch latest updates
        print("Fetching latest updates from the remote repository...")
        subprocess.run(["git", "fetch"], check=True)

        # Check for local changes
        local_changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if local_changes.stdout.strip():
            print("\nYou have local changes:")
            print(local_changes.stdout)
        else:
            print("\nNo local changes detected.")

        # Check if the local branch is behind the remote branch
        print("\nChecking if local branch is behind the remote branch...")
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True)
        if "behind" in status.stdout:
            print("\nYour local branch is behind the remote branch. Run 'git pull' to sync.")
        else:
            print("\nYour local branch is up to date with the remote branch.")

        # Check for untracked files
        print("\nChecking for untracked files...")
        untracked_files = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True)
        if untracked_files.stdout.strip():
            print("\nUntracked files found:")
            print(untracked_files.stdout)
        else:
            print("\nNo untracked files found.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # Update the repo_path to the current workspace directory
    repo_path = "/workspaces/Machine-Learning-for-Identifying-Fraudulent-Calls"
    check_repo_sync(repo_path)
