cd /workspaces/Machine-Learning-for-Identifying-Fraudulent-Calls
touch auto_git_sync.py
import os
import subprocess
import datetime

# Configuration
REPO_DIR = "/workspaces/Machine-Learning-for-Identifying-Fraudulent-Calls"
COMMIT_MESSAGE = f"Automated sync: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def git_sync():
    try:
        # Navigate to the repository
        os.chdir(REPO_DIR)

        # Check for changes
        subprocess.run(["git", "status"], check=True)

        # Add all changes
        subprocess.run(["git", "add", "."], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=True)

        # Push changes
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("✅ Sync completed successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error during sync:", e)

if __name__ == "__main__":
    git_sync()
chmod +x auto_git_sync.py
python3 auto_git_sync.py
