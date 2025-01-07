import os
import subprocess

import yaml

TEST_CONFIG_YAML = "test-config.yaml"
OUTPUT_FILE = "run_tests.sh"


def get_data(yaml_file_name):
    with open(yaml_file_name, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    test_config_yaml_path = os.path.join(os.path.dirname(__file__), TEST_CONFIG_YAML)
    data = get_data(test_config_yaml_path)
    output_file_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    with open(output_file_path, "w") as f:
        print("Started writing to {}".format(OUTPUT_FILE))
        for provider in data["providers"].split(" "):
            for model in data["inference_models"]:
                inference_model, test_file = data["inference_models"][model].split(",")
                f.write(
                    'pytest -v -s -k "{}"  --inference-model="{}" ./llama_stack/providers/tests/inference/{}\n'.format(
                        provider, inference_model, test_file[1:]
                    )
                )
        print("Finished writing to {}".format(OUTPUT_FILE))
        subprocess.run(["chmod", "+x", output_file_path])
        subprocess.run(["bash", output_file_path])


if __name__ == "__main__":
    main()
