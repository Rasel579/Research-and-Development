import os
from datetime import datetime

from github import Github, GithubException
from git import Repo

TOKEN = os.environ['GITHUB_API']
LOGIN = os.environ['GITHUB_LOGIN']

class GithubTool:
    def __init__(self, vk, user_id):
        self.token = TOKEN
        self.login = LOGIN
        self.github = Github(self.token)
        self.vk = vk
        self.user_id = user_id

    def manage_repo_and_push(self, repo_name: str, repo_path: str,
                             branch_name: str = None, commit_message: str = None):
        user = self.github.get_user()

        try:
            repo = self.github.get_repo(f"{self.login}/{repo_name}")
            repo_exists = True
            print(f"✅ Репозиторий {repo_name} найден")
            self.sent_to_vk(f"✅ Репозиторий {repo_name} найден")
        except GithubException:
            repo_exists = False

            print(f"📦 Создаю новый репозиторий {repo_name}...")
            self.sent_to_vk(f"📦 Создаю новый репозиторий {repo_name}...")
            repo = user.create_repo(
                repo_name,
                description=f"Автоматически создан AI-агентом",
                private=False
            )
            print(f"✅ Репозиторий создан: {repo.html_url}")
            self.sent_to_vk(f"✅ Репозиторий создан: {repo.html_url}")

        # 3. Работаем с локальным git
        local_repo = Repo(repo_path)

        if repo_exists:
            if not branch_name:
                branch_name = f"feature/auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            new_branch = local_repo.create_head(branch_name)
            local_repo.heads[branch_name].checkout()
            print(f"🌿 Создана ветка: {branch_name}")
            self.sent_to_vk(f"🌿 Создана ветка: {branch_name}")
            if 'origin' not in [remote.name for remote in local_repo.remotes]:
                local_repo.create_remote('origin', self._get_auth_url(repo_name))

            local_repo.git.add(A=True)
            if not commit_message:
                commit_message = f"Auto commit from {branch_name}"
            local_repo.index.commit(commit_message)

            origin = local_repo.remote('origin')
            origin.push(f'{branch_name}:{branch_name}')
            print(f"✅ Код запушен в ветку {branch_name}")
            self.sent_to_vk(f"✅ Код запушен в ветку {branch_name}")
            pr = repo.create_pull(
                title=f"Auto PR from {branch_name}",
                body=f"Автоматически созданный PR веткой {branch_name}",
                head=branch_name,
                base="main"  # или master
            )
            print(f"✅ Pull Request создан: {pr.html_url}")
            return f"✅ PR создан: {pr.html_url}"

        else:
            # Если репо новый - просто пушим в main
            if 'origin' not in [remote.name for remote in local_repo.remotes]:
                local_repo.create_remote('origin', self._get_auth_url(repo_name))

            local_repo.git.add(A=True)
            if not commit_message:
                commit_message = "Initial commit from AI agent"
            local_repo.index.commit(commit_message)

            origin = local_repo.remote('origin')
            origin.push('main:main')
            print(f"✅ Первый коммит запушен в main")
            return f"✅ Репозиторий создан и код запушен: {repo.html_url}"

    def sent_to_vk(self, message: str):
        try:
            self.vk.messages.send(
                user_id=self.user_id,
                message=message,
                random_id=0
            )
        except Exception as e:
            print(f"Ошибка: {e}")

    def _get_auth_url(self, repo_name: str) -> str:
        return f"https://{self.login}:{self.token}@github.com/{self.login}/{repo_name}.git"
