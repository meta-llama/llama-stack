# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.modules.transforms import Transform


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        message_transform: Transform,
        model_transform: Transform,
    ) -> None:
        self._rows = rows
        self._message_transform = message_transform
        self._model_transform = model_transform

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._rows[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._message_transform(sample)
        if "messages" in transformed_sample:
            validate_messages(transformed_sample["messages"])

        tokenized_dict = self._model_transform(transformed_sample)

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                "model_transform returned the following keys: "
                f"{keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict
