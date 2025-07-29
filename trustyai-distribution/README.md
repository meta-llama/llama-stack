# TrustyAI Distribution Build Instructions

This directory contains the necessary files to build a TrustyAI compatible container image for the llama-stack.

## Prerequisites

- Python >=3.12
- `llama` CLI tool installed: `pip install llama-stack`
- Podman or Docker installed

## Generating the Containerfile

The Containerfile is auto-generated from a template. To generate it:

1. Make sure you have the `llama` CLI tool installed
2. Run the build script from root of this git repo:
   ```bash
   ./trustyai-distribution/build.py
   ```

This will:
- Check for the llama CLI installation
- Generate dependencies using `llama stack build`
- Create a new `Containerfile` with the required dependencies

## Editing the Containerfile

The Containerfile is auto-generated from a template. To edit it, you can modify the template in `trustyai-distribution/Containerfile.in` and run the build script again.
NEVER edit the generated `Containerfile` manually.

## Building the Container Image

Once the Containerfile is generated, you can build the image using either Podman or Docker:

### Using Podman build image for x86_64
```bash
podman build --platform linux/amd64 -f trustyai-distribution/Containerfile -t trustyai .
```

## Notes

- The generated Containerfile should not be modified manually as it will be overwritten the next time you run the build script

## Push the image to a registry

```bash
podman push <build-ID> quay.io/trustyai/llama-stack:trustyai-distribution
```
