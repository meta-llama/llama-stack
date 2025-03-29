# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from fastapi_pagination import Params, paginate

from llama_stack.apis.common.responses import PaginatedResponse


def paginate_records(
    records: List[Dict[str, Any]],
    start_index: Optional[int] = None,
    limit: Optional[int] = None,
) -> PaginatedResponse:
    """Helper function to handle pagination of records consistently across implementations.

    :param records: List of records to paginate
    :param start_index: The starting index (0-based). If None, starts from beginning.
    :param limit: Number of items to return. If None or -1, returns all items.
    :return: PaginatedResponse with the paginated data
    """
    # Handle special case for fetching all rows
    if limit is None or limit == -1:
        return PaginatedResponse(
            data=records,
            total=len(records),
            page=1,
            size=len(records),
            pages=1,
        )

    # Convert start_index/limit to page/size for paginate
    page = (start_index or 0) // limit + 1
    size = limit

    # Use fastapi-pagination for consistent pagination behavior
    params = Params(page=page, size=size)
    paginated = paginate(records, params)
    return PaginatedResponse(
        data=paginated.items,
        total=paginated.total,
        page=paginated.page,
        size=paginated.size,
        pages=paginated.pages,
    )
