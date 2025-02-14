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
from typing import Optional

import fire

# from llama_stack.models.llama.datatypes import *  # noqa: F403
from llama_models.llama3.reference_impl.generation import Llama

THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    ckpt_dir: str,
    module_name: str,
    output_path: str,
    model_parallel_size: Optional[int] = None,
):
    module = importlib.import_module(module_name)
    assert hasattr(module, "usecases"), f"Module {module_name} missing usecases function"
    tokenizer_path = str(THIS_DIR.parent / "llama3/api/tokenizer.model")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=512,
        max_batch_size=1,
        model_parallel_size=model_parallel_size,
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
