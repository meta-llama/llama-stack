# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
from unittest.mock import patch

import pytest

from llama_stack.distribution.access_control.access_control import default_policy
from llama_stack.distribution.datatypes import User
from llama_stack.providers.utils.sqlstore.api import ColumnType
from llama_stack.providers.utils.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.providers.utils.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig, SqliteSqlStoreConfig


def get_postgres_config():
    """Get PostgreSQL configuration if tests are enabled."""
    return PostgresSqlStoreConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        db=os.environ.get("POSTGRES_DB", "llamastack"),
        user=os.environ.get("POSTGRES_USER", "llamastack"),
        password=os.environ.get("POSTGRES_PASSWORD", "llamastack"),
    )


def get_sqlite_config():
    """Get SQLite configuration with temporary database."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_file.close()
    return SqliteSqlStoreConfig(db_path=tmp_file.name), tmp_file.name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_config",
    [
        pytest.param(
            ("postgres", get_postgres_config),
            marks=pytest.mark.skipif(
                not os.environ.get("ENABLE_POSTGRES_TESTS"),
                reason="PostgreSQL tests require ENABLE_POSTGRES_TESTS environment variable",
            ),
            id="postgres",
        ),
        pytest.param(("sqlite", get_sqlite_config), id="sqlite"),
    ],
)
@patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_json_comparison(mock_get_authenticated_user, backend_config):
    """Test that JSON column comparisons work correctly for both PostgreSQL and SQLite"""
    backend_name, config_func = backend_config

    # Handle different config types
    if backend_name == "postgres":
        config = config_func()
        cleanup_path = None
    else:  # sqlite
        config, cleanup_path = config_func()

    try:
        base_sqlstore = SqlAlchemySqlStoreImpl(config)
        authorized_store = AuthorizedSqlStore(base_sqlstore)

        # Create test table
        table_name = f"test_json_comparison_{backend_name}"
        await authorized_store.create_table(
            table=table_name,
            schema={
                "id": ColumnType.STRING,
                "data": ColumnType.STRING,
            },
        )

        try:
            # Test with no authenticated user (should handle JSON null comparison)
            mock_get_authenticated_user.return_value = None

            # Insert some test data
            await authorized_store.insert(table_name, {"id": "1", "data": "public_data"})

            # Test fetching with no user - should not error on JSON comparison
            result = await authorized_store.fetch_all(table_name, policy=default_policy())
            assert len(result.data) == 1
            assert result.data[0]["id"] == "1"
            assert result.data[0]["access_attributes"] is None

            # Test with authenticated user
            test_user = User("test-user", {"roles": ["admin"]})
            mock_get_authenticated_user.return_value = test_user

            # Insert data with user attributes
            await authorized_store.insert(table_name, {"id": "2", "data": "admin_data"})

            # Fetch all - admin should see both
            result = await authorized_store.fetch_all(table_name, policy=default_policy())
            assert len(result.data) == 2

            # Test with non-admin user
            regular_user = User("regular-user", {"roles": ["user"]})
            mock_get_authenticated_user.return_value = regular_user

            # Should only see public record
            result = await authorized_store.fetch_all(table_name, policy=default_policy())
            assert len(result.data) == 1
            assert result.data[0]["id"] == "1"

            # Test the category missing branch: user with multiple attributes
            multi_user = User("multi-user", {"roles": ["admin"], "teams": ["dev"]})
            mock_get_authenticated_user.return_value = multi_user

            # Insert record with multi-user (has both roles and teams)
            await authorized_store.insert(table_name, {"id": "3", "data": "multi_user_data"})

            # Test different user types to create records with different attribute patterns
            # Record with only roles (teams category will be missing)
            roles_only_user = User("roles-user", {"roles": ["admin"]})
            mock_get_authenticated_user.return_value = roles_only_user
            await authorized_store.insert(table_name, {"id": "4", "data": "roles_only_data"})

            # Record with only teams (roles category will be missing)
            teams_only_user = User("teams-user", {"teams": ["dev"]})
            mock_get_authenticated_user.return_value = teams_only_user
            await authorized_store.insert(table_name, {"id": "5", "data": "teams_only_data"})

            # Record with different roles/teams (shouldn't match our test user)
            different_user = User("different-user", {"roles": ["user"], "teams": ["qa"]})
            mock_get_authenticated_user.return_value = different_user
            await authorized_store.insert(table_name, {"id": "6", "data": "different_user_data"})

            # Now test with the multi-user who has both roles=admin and teams=dev
            mock_get_authenticated_user.return_value = multi_user
            result = await authorized_store.fetch_all(table_name, policy=default_policy())

            # Should see:
            # - public record (1) - no access_attributes
            # - admin record (2) - user matches roles=admin, teams missing (allowed)
            # - multi_user record (3) - user matches both roles=admin and teams=dev
            # - roles_only record (4) - user matches roles=admin, teams missing (allowed)
            # - teams_only record (5) - user matches teams=dev, roles missing (allowed)
            # Should NOT see:
            # - different_user record (6) - user doesn't match roles=user or teams=qa
            expected_ids = {"1", "2", "3", "4", "5"}
            actual_ids = {record["id"] for record in result.data}
            assert actual_ids == expected_ids, f"Expected to see records {expected_ids} but got {actual_ids}"

            # Verify the category missing logic specifically
            # Records 4 and 5 test the "category missing" branch where one attribute category is missing
            category_test_ids = {record["id"] for record in result.data if record["id"] in ["4", "5"]}
            assert category_test_ids == {"4", "5"}, (
                f"Category missing logic failed: expected 4,5 but got {category_test_ids}"
            )

        finally:
            # Clean up records
            for record_id in ["1", "2", "3", "4", "5", "6"]:
                try:
                    await base_sqlstore.delete(table_name, {"id": record_id})
                except Exception:
                    pass

    finally:
        # Clean up temporary SQLite database file if needed
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass
