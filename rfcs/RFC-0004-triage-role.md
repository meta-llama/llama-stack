# The Llama Stack API

**Author:**

* Red Hat: @franciscojavierarceo @nathan-weinberg

## Summary

The Llama Stack project has grown substantially. To reduce maintainer burden, we propose creating a [Triage-level](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization#repository-roles-for-organizations) role in Llama Stack.

## Motivation

The project has had a large increase in the number of contributors, which has led to a growing backlog of requests. Contributors looking to help refine that backlog, unfortunately, need to get in touch with a maintainer to help, further adding to the maintainer's load.

By adding a Triage role, we hope to provide a way for established contributors to help the maintainers manage the requests from the community.

## Permissions for the Triage-role

The incremental permissions a Triage role has above a Read only role are:

1. Apply/dismiss labels
1. Close, reopen, and assign all issues and pull requests
1. Apply milestones
1. Mark duplicate issues and pull requests
1. Request pull request reviews
1. Hide anyone's comments
1. Move a discussion to a different category
1. Lock and unlock discussions
1. Individually convert issues to discussions
1. Delete a discussion

Importantly, the Triage-role cannot:

1. Approve or request changes to a pull request with required reviews
1. Apply suggested changes to pull requests
1. Edit wikis in private repositories
1. Create, edit, run, re-run, and cancel GitHub Actions workflows
1. Create and edit releases

And other important items outlined more in depth in the [GitHub documentation](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization#permissions-for-each-role).

## Nomination Process for Triage-role
The process for nomination for the triage role should be simple and at the discretion of the maintainers.

## Example

We tested this functionality using the @feast-dev repository and have provided screenshots outlining how to make this change.

Step 1:
![Figure 1: Select Repository Settings](./_static/triage-role-config-1.png)

Step 2:
![Figure 2: Invite Outside Collaborator](./_static/triage-role-config-2.png)

Step 3:
![Figure 3: Select Triage Role](./_static/triage-role-config-3.png)

Step 4:
![Figure 4: User Receives Triage Role](./_static/triage-role-config-4.png)


## Thank you

Thank you in advance for your feedback and support and we look forward to collaborating on this great project!

Cheers!
