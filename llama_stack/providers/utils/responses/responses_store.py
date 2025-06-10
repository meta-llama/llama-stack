# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.agents import (
    Order,
)
from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIResponseInput,
    OpenAIResponseObject,
    OpenAIResponseObjectWithInput,
)
from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR

from ..sqlstore.api import ColumnDefinition, ColumnType
from ..sqlstore.sqlstore import SqliteSqlStoreConfig, SqlStoreConfig, sqlstore_impl


class ResponsesStore:
    def __init__(self, sql_store_config: SqlStoreConfig):
        if not sql_store_config:
            sql_store_config = SqliteSqlStoreConfig(
                db_path=(RUNTIME_BASE_DIR / "sqlstore.db").as_posix(),
            )
        self.sql_store = sqlstore_impl(sql_store_config)

    async def initialize(self):
        """Create the necessary tables if they don't exist."""
        await self.sql_store.create_table(
            "openai_responses",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "response_object": ColumnType.JSON,
                "model": ColumnType.STRING,
            },
        )

    async def store_response_object(
        self, response_object: OpenAIResponseObject, input: list[OpenAIResponseInput]
    ) -> None:
        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]

        await self.sql_store.insert(
            "openai_responses",
            {
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "response_object": data,
            },
        )

    async def list_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """
        List responses from the database.

        :param after: The ID of the last response to return.
        :param limit: The maximum number of responses to return.
        :param model: The model to filter by.
        :param order: The order to sort the responses by.
        """
        # TODO: support after
        if after:
            raise NotImplementedError("After is not supported for SQLite")
        if not order:
            order = Order.desc

        rows = await self.sql_store.fetch_all(
            "openai_responses",
            where={"model": model} if model else None,
            order_by=[("created_at", order.value)],
            limit=limit,
        )

        data = [OpenAIResponseObjectWithInput(**row["response_object"]) for row in rows]
        return ListOpenAIResponseObject(
            data=data,
            # TODO: implement has_more
            has_more=False,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_response_object(self, response_id: str) -> OpenAIResponseObjectWithInput:
        row = await self.sql_store.fetch_one("openai_responses", where={"id": response_id})
        if not row:
            raise ValueError(f"Response with id {response_id} not found") from None
        return OpenAIResponseObjectWithInput(**row["response_object"])

    async def list_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """
        List input items for a given response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        """
        # TODO: support after/before pagination
        if after or before:
            raise NotImplementedError("After/before pagination is not supported yet")
        if include:
            raise NotImplementedError("Include is not supported yet")

        response_with_input = await self.get_response_object(response_id)
        input_items = response_with_input.input

        if order == Order.desc:
            input_items = list(reversed(input_items))

        if limit is not None and len(input_items) > limit:
            input_items = input_items[:limit]

        return ListOpenAIResponseInputItem(data=input_items)
