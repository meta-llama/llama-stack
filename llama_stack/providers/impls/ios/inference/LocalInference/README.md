# LocalInference

LocalInference provides a local inference implementation powered by [executorch](https://github.com/pytorch/executorch/).

Llama Stack currently supports on-device inference for iOS with Android coming soon. You can run on-device inference on Android today using [executorch](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/LlamaDemo), PyTorch’s on-device inference library.

## Installation

We're working on making LocalInference easier to set up. For now, you'll need to import it via `.xcframework`:

1. Clone the executorch submodule in this repo and its dependencies: `git submodule update --init --recursive`
1. Install [Cmake](https://cmake.org/) for the executorch build`
1. Drag `LocalInference.xcodeproj` into your project
1. Add `LocalInference` as a framework in your app target
1. Add a package dependency on https://github.com/pytorch/executorch (branch latest)
1. Add all the kernels / backends from executorch (but not exectuorch itself!) as frameworks in your app target:
    - backend_coreml
    - backend_mps
    - backend_xnnpack
    - kernels_custom
    - kernels_optimized
    - kernels_portable
    - kernels_quantized
1. In "Build Settings" > "Other Linker Flags" > "Any iOS Simulator SDK", add:
    ```
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_optimized-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_custom-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_quantized-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_xnnpack-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_coreml-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_mps-simulator-release.a
    ```

1. In "Build Settings" > "Other Linker Flags" > "Any iOS SDK", add:

    ```
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_optimized-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_custom-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libkernels_quantized-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_xnnpack-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_coreml-simulator-release.a
    -force_load
    $(BUILT_PRODUCTS_DIR)/libbackend_mps-simulator-release.a
    ```

## Preparing a model

1. Prepare a `.pte` file [following the executorch docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md#step-2-prepare-model)
2. Bundle the `.pte` and `tokenizer.model` file into your app

## Using LocalInference

1. Instantiate LocalInference with a DispatchQueue. Optionally, pass it into your agents service:

```swift
  init () {
    runnerQueue = DispatchQueue(label: "org.meta.llamastack")
    inferenceService = LocalInferenceService(queue: runnerQueue)
    agentsService = LocalAgentsService(inference: inferenceService)
  }
```

2. Before making any inference calls, load your model from your bundle:

```swift
let mainBundle = Bundle.main
inferenceService.loadModel(
    modelPath: mainBundle.url(forResource: "llama32_1b_spinquant", withExtension: "pte"),
    tokenizerPath: mainBundle.url(forResource: "tokenizer", withExtension: "model"),
    completion: {_ in } // use to handle load failures
)
```

3. Make inference calls (or agents calls) as you normally would with LlamaStack:

```
for await chunk in try await agentsService.initAndCreateTurn(
    messages: [
    .UserMessage(Components.Schemas.UserMessage(
        content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + text),
        role: .user))
    ]
) {
```

## Troubleshooting

If you receive errors like "missing package product" or "invalid checksum", try cleaning the build folder and resetting the Swift package cache:

(Opt+Click) Product > Clean Build Folder Immediately

```
rm -rf \
  ~/Library/org.swift.swiftpm \
  ~/Library/Caches/org.swift.swiftpm \
  ~/Library/Caches/com.apple.dt.Xcode \
  ~/Library/Developer/Xcode/DerivedData
```
