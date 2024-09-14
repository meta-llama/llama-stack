from llama_toolchain.core.distribution_registry import *
import json

import fire
import yaml
from llama_toolchain.common.serialize import EnumEncoder


def main():
    for d in available_distribution_specs():
        file_path = "./configs/distributions/distribution_registry/{}.yaml".format(
            d.distribution_type
        )

        with open(file_path, "w") as f:
            to_write = json.loads(json.dumps(d.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))


if __name__ == "__main__":
    fire.Fire(main)
