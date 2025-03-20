#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import importlib
from pathlib import Path

import fire

from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.providers.inline.inference.meta_reference.config import MetaReferenceInferenceConfig
from llama_stack.providers.inline.inference.meta_reference.llama3.generation import Llama3

THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    model_id: str,
    checkpoint_dir: str,
    module_name: str,
    output_path: str,
):
    module = importlib.import_module(module_name)
    assert hasattr(module, "usecases"), f"Module {module_name} missing usecases function"

    config = MetaReferenceInferenceConfig(
        model=model_id,
        max_seq_len=512,
        max_batch_size=1,
        checkpoint_dir=checkpoint_dir,
    )
    llama_model = resolve_model(model_id)
    if not llama_model:
        raise ValueError(f"Model {model_id} not found")
    generator = Llama3.build(
        config=config,
        model_id=model_id,
        llama_model=llama_model,
    )

    use_cases = module.usecases()
    text = ""
    for u in use_cases:
        if isinstance(u, str):
            use_case_text = f"\n{u}\n"
        else:
            use_case_text = u.to_text(generator)

        text += use_case_text
        print(use_case_text)

    text += "Thank You!\n"

    with open(output_path, "w") as f:
        f.write(text)


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
