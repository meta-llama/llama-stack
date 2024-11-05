from inline import LlamaStackInline
from llama_stack.apis.inference.inference import Inference

from llama_stack.providers.datatypes import *  # noqa: F403


async def main():
    inline = LlamaStackInline("/home/dalton/.llama/builds/conda/nov5-run.yaml")
    await inline.initialize()
    print(inline.impls)


# Run the main function
import asyncio

asyncio.run(main())
