# Enforcing contribution rules for repository stability with a merging bot

**Authors:**

* Red Hat: @leseb

## Summary

To ensure the stability, security, and maintainability of the project, we propose enforcing the following contribution rules. These guidelines help maintain code quality, prevent accidental regressions, and improve collaboration across contributors.

We recognize that many of these rules may already be considered common sense by experienced contributors. However, making them explicit and automating their enforcement ensures consistency across all contributors, avoids misunderstandings, and streamlines the development process. The goal is not to restrict contributions but to create a smoother workflow that benefits everyone.

Rather than relying solely on manual enforcement, we propose using a merging bot to handle these rules automatically, reducing the risk of human error and keeping the process efficient.

## Proposed Rules & Automated Enforcement

1. Do not push directly to main
[Branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule) settings will prevent direct pushes to main, requiring all changes to go through pull requests (PRs).

2. Do not self-merge
The merging bot will ensure that a PR has at least one approval from another maintainer before merging.

3. Do not merge if CI is failing
The bot will block merging if the CI checks do not pass, ensuring that only validated code is integrated. Maintainers with write access can override the bot and merge the PR anyway if needed.

4. Do not use the main repository for development, use a fork instead
Documentation will guide contributors on forking best practices, and a workflow can optionally reject PRs not coming from a fork.

5. Require code to be up to date with main before merging
The bot will check if the PR is outdated compared to main and require a rebase or merge with the latest main branch before allowing the merge.

By automating these rules through a merging bot, we ensure a consistent, high-quality, and efficient development process with minimal manual oversight.

## Sample bot configuration

[Mergify](https://mergify.com/) is a merging bot that can be configured to enforce the proposed rules.

Here a sample configuration::

```yaml
pull_request_rules:
- name: auto-merge
  conditions:
    - "#approved-reviews-by>=1"
    - check-success=pre-commit
    - check-success="Facebook CLA Check"
    - check-success="unit-tests"
    - check-success="integration-tests"

  actions:
    merge:
      method: squash
      commit_message: title+body
```

The file can be summarized as requiring at least one approval and CI being green before squashing and merging automatically.

## Addressing Concerns About the Merging Bot

Some maintainers may worry that a merging bot will slow down development by adding extra steps to the workflow. However, this is not the case - the bot actually helps accelerate the process in several ways:

* Prevents last-minute firefighting: by ensuring only properly reviewed and validated code gets merged, the bot reduces the chances of breaking main, which can save significant time spent debugging regressions.
* Eliminates manual enforcement: instead of relying on maintainers to remind contributors of best practices, the bot handles rule enforcement automatically, making the process smoother and more predictable.
* Reduces merge conflicts: since PRs must be up to date with main before merging, contributors encounter and resolve conflicts before merging, preventing disruptive post-merge conflicts.
* Speeds up CI troubleshooting: by blocking PRs with failing CI, issues are addressed before merging, rather than surfacing later and forcing costly rollbacks.

The merging bot is not an obstacle - it’s a tool designed to make development faster and more reliable by keeping the codebase clean and reducing friction for contributors.
