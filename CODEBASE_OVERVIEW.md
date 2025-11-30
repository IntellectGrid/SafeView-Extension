# SafeView Extension Codebase Overview

This document provides a high-level overview of the SafeView Chrome Extension codebase, explaining its architecture, key files, and how the content filtering works.

## Architecture

SafeView is a **Manifest V3** Chrome Extension that uses client-side AI to detect and blur NSFW content and specific face/gender attributes in real-time.

### Key Components

1.  **Content Script (`content.js`)**: Runs on the web page. It observes the DOM for images and videos, sends them for analysis, and applies blur filters based on the results.
2.  **Offscreen Document (`offscreen.html` / `offscreen.js`)**: A hidden document that handles the heavy AI processing. It uses `onnxruntime-web` and `Human.js` to run inference without blocking the main browser thread or violating CSP.
3.  **Service Worker (`background.js`)**: Orchestrates the extension's lifecycle, creates the offscreen document, and handles context menus.
4.  **AI Modules (`modules/detector.js`)**: Encapsulates the machine learning models and inference logic.

## File Descriptions

### Root Directory (`src/`)

-   **`manifest.json`**: The extension's configuration file. Defines permissions (`offscreen`, `storage`, `activeTab`), content scripts, and background worker.
-   **`background.js`**: The service worker.
    -   Initializes the extension on installation.
    -   Creates the **Offscreen Document** (`chrome.offscreen.createDocument`).
    -   Manages context menu items (e.g., "Enable for this video").
-   **`content.js`**: The entry point for web pages.
    -   Initializes `MutationObserver` to track new elements.
    -   Communicates with the offscreen document to request image analysis.
    -   Applies CSS filters to blur content.
-   **`offscreen.html`**: The HTML container for the offscreen document. Loads `ort.min.js` (ONNX Runtime) locally.
-   **`offscreen.js`**: The script running inside the offscreen document.
    -   Initializes the `Detector` class.
    -   Manages a `Queue` to process images sequentially.
    -   Receives messages from `content.js`, runs inference, and returns results.
    -   **Key Fix**: Configures `ort.env.wasm.wasmPaths` to load WASM files from the extension's `assets/` directory.

### Modules (`src/modules/`)

-   **`detector.js`**: The core AI logic.
    -   **`initHuman()`**: Loads `Human.js` for face/gender detection.
    -   **`initNsfwModel()`**: Loads the custom NSFW ONNX model. Handles caching in IndexedDB.
    -   **`nsfwModelClassify()`**: Preprocesses images (resizing, NCHW transposition) and runs inference.
    -   **`humanModelClassify()`**: Runs face detection.
    -   **Key Fixes**: Includes initialization locks to prevent race conditions and NCHW input shape correction.
-   **`settings.js`**: Manages user preferences.
    -   Handles storage retrieval and updates.
    -   Provides helper methods like `shouldDetectImages()`, `getStrictness()`, etc.
-   **`queues.js`**: Implements a processing queue.
    -   Ensures images are processed one by one to prevent GPU memory exhaustion.
    -   Prioritizes requests and manages concurrency.
-   **`helpers.js`**: Utility functions.
    -   `loadImage()` / `loadVideo()`: Helper wrappers for loading media.
    -   `calcResize()`: Resizes images to optimize inference speed.
    -   `getCanvas()`: Creates offscreen canvases for image manipulation.

## Workflows

### 1. Image Detection Pipeline

1.  **Observation**: `content.js` detects a new `<img>` element via `MutationObserver`.
2.  **Request**: It sends a message `{ type: "imageDetection", image: src }` to the runtime.
3.  **Routing**: The message is received by `offscreen.js`.
4.  **Queueing**: The request is added to the `Queue`.
5.  **Processing**:
    -   `offscreen.js` converts the image URL to an `ImageData` object.
    -   It calls `detector.nsfwModelClassify()`.
    -   The image is resized to 224x224 and normalized.
    -   The ONNX model runs inference.
6.  **Result**: The classification (e.g., "nsfw", "face", or false) is sent back to `content.js`.
7.  **Action**: `content.js` applies a blur filter if the result matches user settings.

### 2. Video Detection Pipeline

1.  **Frame Extraction**: `content.js` extracts frames from `<video>` elements at regular intervals.
2.  **Request**: Sends `{ type: "videoDetection", frame: data }`.
3.  **Throttling**: `offscreen.js` checks if the queue is busy. If so, it **drops the frame** to prevent lag.
4.  **Inference**: If processed, it follows the same path as image detection.

## Setup & Building

-   **Dependencies**: `onnxruntime-web`, `human`.
-   **Assets**: Model files (`.onnx`, `.wasm`, `.mjs`) must be present in `src/assets/` for the extension to function without external CDN calls (MV3 compliance).
