import os
import subprocess

COMMIT_TYPES = [
    "feat (Adding a new feature)",
    "fix (Fixing a bug)",
    "docs (Documentation changes)",
    "style (Code style changes)",
    "refactor (Code restructuring without changing functionality)",
    "test (Adding or modifying tests)",
    "chore (Routine tasks like package updates)",
    "build (Changes to the build system or dependencies)",
    "ci (Continuous integration changes)",
    "perf (Performance improvements)",
    "revert (Reverting a previous commit)"
]

def get_commit_message():
    print("\nSelect a commit type:")
    for i, commit_type in enumerate(COMMIT_TYPES, 1):
        print(f"{i}. {commit_type}")

    while True:
        try:
            commit_type_choice = int(input("\nEnter the number corresponding to your commit type: ").strip())
            if commit_type_choice < 1 or commit_type_choice > len(COMMIT_TYPES):
                raise ValueError("Invalid number")
            
            commit_type = COMMIT_TYPES[commit_type_choice - 1].split()[0]
            break
        except ValueError:
            print("\nInvalid choice. Please enter a number between 1 and 11.")

    scope = input("\nEnter the scope (e.g., 'scripts', 'project', 'readme', 'structure', 'sync'): ").strip()
    description = input("\nEnter a short description of the changes: ").strip()

    # Format the final commit message
    commit_message = f"{commit_type}({scope}): {description}"
    print(f"\nGenerated commit message: **{commit_message}**")
    
    # Confirm the final commit message
    confirm = input("\nDo you want to use this commit message? (yes/no): ").strip().lower()
    if confirm == "yes":
        return commit_message
    else:
        print("\nCommit message discarded. Please try again.")
        return get_commit_message()

def check_repo_sync(repo_path):
    try:
        # Check if the directory is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            print("\nNot a valid Git repository. Please initialize a git repository.")
            return
        
        # Navigate to the repository
        os.chdir(repo_path)

        # Fetch latest updates
        print("\nFetching latest updates from the remote repository...")
        subprocess.run(["git", "fetch"], check=True)

        # Check for local changes
        print("\nChecking for local changes...")
        local_changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if local_changes.stdout.strip():
            print("\nLocal changes detected:")
            print(local_changes.stdout)

            # Prompt user to commit changes
            commit = input("\nDo you want to commit and push these changes? (yes/no): ").strip().lower()
            if commit == "yes":
                commit_message = get_commit_message()
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                subprocess.run(["git", "push"], check=True)
                print("\nChanges committed and pushed successfully.")
            else:
                print("\nLocal changes were not committed.")
        else:
            print("\nNo local changes detected.")

        # Check if the local branch is behind the remote branch
        print("\nChecking if the local branch is behind the remote branch...")
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True)
        if "behind" in status.stdout:
            print("\nYour local branch is behind the remote branch. Run 'git pull' to sync.")
            pull_sync = input("Do you want to pull the latest changes? (yes/no): ").strip().lower()
            if pull_sync == "yes":
                subprocess.run(["git", "pull"], check=True)
                print("\nLocal branch synced with the remote branch.")
            else:
                print("\nLocal branch not synced.")
        else:
            print("\nYour local branch is up to date with the remote branch.")

        # Check for untracked files
        print("\nChecking for untracked files...")
        untracked_files = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True)
        if untracked_files.stdout.strip():
            print("\nUntracked files found:")
            print(untracked_files.stdout)
            add_untracked = input("\nDo you want to add and push these untracked files? (yes/no): ").strip().lower()
            if add_untracked == "yes":
                commit_message = get_commit_message()
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                subprocess.run(["git", "push"], check=True)
                print("\nUntracked files committed and pushed successfully.")
            else:
                print("\nUntracked files were not added.")
        else:
            print("\nNo untracked files found.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # Update the repo_path to the current workspace directory
    repo_path = "/workspaces/Machine-Learning-for-Identifying-Fraudulent-Calls"
    check_repo_sync(repo_path)
