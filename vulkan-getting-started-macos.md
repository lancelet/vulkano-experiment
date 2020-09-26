# Vulkan Getting Started

1. Install [Homebrew](https://brew.sh/).

2. Add a Homebrew Cask which includes the Vulkan SDK for MacOS:
    ```sh
    brew tap apenngrace/homebrew-vulkan
    ```

3. Install the Vulkan SDK from the `homebrew-vulkan` Cask:
    ```sh
    brew tap apenngrace/homebrew-vulkan
    ```

4. Set the required environment variables. Exact paths my vary. eg:
    ```sh
    export VULKAN_SDK='/usr/local/Caskroom/vulkan-sdk/1.2.148.0/macOS'
    export PATH="$VULKAN_SDK/bin:$PATH"
    export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH"
    export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
    export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
    ```

5. Install `cmake` for the `vulkano-shaders` crate:
    ```sh
    brew install cmake
    ```