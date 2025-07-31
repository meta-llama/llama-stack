# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.inference import (
    ListOpenAIChatCompletionResponse,
    OpenAIChatCompletion,
    OpenAICompletionWithInputMessages,
    OpenAIMessageParam,
    Order,
)
from llama_stack.core.datatypes import AccessRule
from llama_stack.core.utils.config_dirs import RUNTIME_BASE_DIR

from ..sqlstore.api import ColumnDefinition, ColumnType
from ..sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ..sqlstore.sqlstore import SqliteSqlStoreConfig, SqlStoreConfig, sqlstore_impl


class InferenceStore:
    def __init__(self, sql_store_config: SqlStoreConfig, policy: list[AccessRule]):
        if not sql_store_config:
            sql_store_config = SqliteSqlStoreConfig(
                db_path=(RUNTIME_BASE_DIR / "sqlstore.db").as_posix(),
            )
        self.sql_store_config = sql_store_config
        self.sql_store = None
        self.policy = policy

    async def initialize(self):
        """Create the necessary tables if they don't exist."""
        self.sql_store = AuthorizedSqlStore(sqlstore_impl(self.sql_store_config))
        await self.sql_store.create_table(
            "chat_completions",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created": ColumnType.INTEGER,
                "model": ColumnType.STRING,
                "choices": ColumnType.JSON,
                "input_messages": ColumnType.JSON,
            },
        )

    async def store_chat_completion(
        self, chat_completion: OpenAIChatCompletion, input_messages: list[OpenAIMessageParam]
    ) -> None:
        if not self.sql_store:
            raise ValueError("Inference store is not initialized")

        data = chat_completion.model_dump()

        await self.sql_store.insert(
            table="chat_completions",
            data={
                "id": data["id"],
                "created": data["created"],
                "model": data["model"],
                "choices": data["choices"],
                "input_messages": [message.model_dump() for message in input_messages],
            },
        )

    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        """
        List chat completions from the database.

        :param after: The ID of the last chat completion to return.
        :param limit: The maximum number of chat completions to return.
        :param model: The model to filter by.
        :param order: The order to sort the chat completions by.
        """
        if not self.sql_store:
            raise ValueError("Inference store is not initialized")

        if not order:
            order = Order.desc

        where_conditions = {}
        if model:
            where_conditions["model"] = model

        paginated_result = await self.sql_store.fetch_all(
            table="chat_completions",
            where=where_conditions if where_conditions else None,
            order_by=[("created", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
            policy=self.policy,
        )

        data = [
            OpenAICompletionWithInputMessages(
                id=row["id"],
                created=row["created"],
                model=row["model"],
                choices=row["choices"],
                input_messages=row["input_messages"],
            )
            for row in paginated_result.data
        ]
        return ListOpenAIChatCompletionResponse(
            data=data,
            has_more=paginated_result.has_more,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        if not self.sql_store:
            raise ValueError("Inference store is not initialized")

        row = await self.sql_store.fetch_one(
            table="chat_completions",
            where={"id": completion_id},
            policy=self.policy,
        )

        if not row:
            # SecureSqlStore will return None if record doesn't exist OR access is denied
            # This provides security by not revealing whether the record exists
            raise ValueError(f"Chat completion with id {completion_id} not found") from None

        return OpenAICompletionWithInputMessages(
            id=row["id"],
            created=row["created"],
            model=row["model"],
            choices=row["choices"],
            input_messages=row["input_messages"],
        )
