// detector.js
// This module exports detector functions and variables
const nsfwOnnxUrl = chrome.runtime.getURL("src/assets/models/NSFWX/model.onnx");

const HUMAN_CONFIG = {
    modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
    backend: "humangl",
    // debug: true,
    cacheSensitivity: 0.9,
    warmup: "none",
    async: true,
    filter: {
        enabled: false,
        // width: 224,
        // height: 224,
    },
    face: {
        enabled: true,
        iris: { enabled: false },
        mesh: { enabled: false },
        emotion: { enabled: false },
        detector: {
            modelPath: "blazeface.json",
            maxDetected: 2,
            minConfidence: 0.25,
        },
        description: {
            enabled: true,
            modelPath: "faceres.json",
        },
    },
    body: {
        enabled: false,
    },
    hand: {
        enabled: false,
    },
    gesture: {
        enabled: false,
    },
    object: {
        enabled: false,
    },
};

const NSFW_CONFIG = {
    size: 224,
    tfScalar: 255,
    topK: 3,
    skipTime: 4000,
    skipFrames: 99,
    cacheSensitivity: 0.9,
};

const getNsfwClasses = (factor = 0) => {
    // factor is a number between 0 and 1
    // it's used to increase the threshold for nsfw classes
    // the numbers are based on trial and error
    return {
        0: {
            className: "Drawing",
            nsfw: false,
            thresh: 0.5,
        },
        1: {
            className: "Hentai",
            nsfw: true,
            thresh: 0.5 + (1 - factor) * 0.5, // decrease the factor to make it less strict
        },
        2: {
            className: "Neutral",
            nsfw: false,
            thresh: 0.5 + factor * 0.5, // increase the factor to make it less strict
        },
        3: {
            className: "Porn",
            nsfw: true,
            thresh: 0.1 + (1 - factor) * 0.4, // decrease the factor to make it less strict
        },
        4: {
            className: "Sexy",
            nsfw: true,
            thresh: 0.1 + (1 - factor) * 0.95, // decrease the factor to make it less strict
        },
    };
};

class Detector {
    constructor() {
        this._human = null;
        this._nsfwModel = null;
        this._initializingHuman = false;
        this._initializingNsfw = false;
        this.nsfwCache = {
            predictions: [],
            timestamp: 0,
            skippedFrames: 0,
            lastInputTensor: null,
        };
    }

    get human() {
        return this._human;
    }

    get nsfwModel() {
        return this._nsfwModel;
    }

    initHuman = async () => {
        if (this._human) return;
        if (this._initializingHuman) {
            // Wait for initialization to complete
            while (this._initializingHuman) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return;
        }

        this._initializingHuman = true;
        try {
            this._human = new Human.Human(HUMAN_CONFIG);
            await this._human.load();
            this._human.tf.enableProdMode();
            // warmup the model
            const tensor = this._human.tf.zeros([1, 224, 224, 3]);
            await this._human.detect(tensor);
            this._human.tf.dispose(tensor);
            console.log("HB==Human model warmed up");
        } finally {
            this._initializingHuman = false;
        }
    };

    humanModelClassify = async (tensor, needToResize) => {
        if (!this._human) await this.initHuman();
        return new Promise((resolve, reject) => {
            const promise = needToResize
                ? this._human.detect(tensor, {
                    filter: {
                        enabled: true,
                        width: needToResize?.newWidth,
                        height: needToResize?.newHeight,
                    },
                })
                : this._human.detect(tensor);
            promise
                .then((res) => {
                    resolve(res);
                })
                .catch((err) => {
                    reject(err);
                });
        });
    };

    initNsfwModel = async () => {
        if (this._nsfwModel) return;
        if (this._initializingNsfw) {
            // Wait for initialization to complete
            while (this._initializingNsfw) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return;
        }

        this._initializingNsfw = true;
        try {
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error("ONNX Runtime (ort) is not available. Make sure ONNX Runtime Web is loaded.");
            }

            // Try to load from cache first
            const cachedSession = await this._loadCachedOnnxModel("nsfw-model");
            if (cachedSession) {
                this._nsfwModel = cachedSession;
                console.log("HB==NSFW model loaded from cache");
            } else {
                // Load from URL
                this._nsfwModel = await ort.InferenceSession.create(nsfwOnnxUrl);

                // Cache for next time
                await this._cacheOnnxModel("nsfw-model", nsfwOnnxUrl);
                console.log("HB==NSFW model loaded from URL and cached");
            }

            // Warmup the model with dummy data
            try {
                const warmupData = new Float32Array(1 * 3 * 224 * 224).fill(0);
                const warmupTensor = new ort.Tensor("float32", warmupData, [1, 3, 224, 224]);

                // Get actual input/output names from model
                const inputName = this._nsfwModel.inputNames[0];
                const feeds = { [inputName]: warmupTensor };
                await this._nsfwModel.run(feeds);

                console.log("HB==NSFW model warmed up");
            } catch (warmupError) {
                console.warn("HB==Warmup failed (non-critical):", warmupError);
            }
        } catch (error) {
            console.error("HB==Failed to initialize NSFW model:", error);
            throw error;
        } finally {
            this._initializingNsfw = false;
        }
    };

    // Helper method: Load ONNX model from IndexedDB cache
    _loadCachedOnnxModel = async (modelName) => {
        return new Promise((resolve) => {
            try {
                const request = indexedDB.open("nsfw-models", 1);

                request.onerror = () => {
                    console.log("HB==Cache read failed, will load from URL");
                    resolve(null);
                };

                request.onupgradeneeded = (event) => {
                    const db = event.target.result;
                    if (!db.objectStoreNames.contains("models")) {
                        db.createObjectStore("models");
                    }
                };

                request.onsuccess = (event) => {
                    try {
                        const db = event.target.result;
                        // Check if object store exists before trying to access it
                        if (!db.objectStoreNames.contains("models")) {
                            console.log("HB==Object store not found, will load from URL");
                            db.close();
                            resolve(null);
                            return;
                        }

                        const transaction = db.transaction(["models"], "readonly");
                        const store = transaction.objectStore("models");
                        const getRequest = store.get(modelName);

                        getRequest.onsuccess = () => {
                            if (getRequest.result) {
                                ort.InferenceSession.create(getRequest.result.buffer)
                                    .then(session => {
                                        db.close();
                                        resolve(session);
                                    })
                                    .catch(() => {
                                        db.close();
                                        resolve(null);
                                    });
                            } else {
                                db.close();
                                resolve(null);
                            }
                        };
                        getRequest.onerror = () => {
                            db.close();
                            resolve(null);
                        };
                    } catch (e) {
                        console.error("HB==Error accessing cache:", e);
                        resolve(null);
                    }
                };
            } catch (e) {
                console.error("HB==Cache error:", e);
                resolve(null);
            }
        });
    };

    // Helper method: Cache ONNX model to IndexedDB
    _cacheOnnxModel = async (modelName, modelUrl) => {
        return new Promise((resolve) => {
            try {
                fetch(modelUrl)
                    .then(response => response.arrayBuffer())
                    .then(buffer => {
                        const request = indexedDB.open("nsfw-models", 1);

                        request.onupgradeneeded = (event) => {
                            const db = event.target.result;
                            if (!db.objectStoreNames.contains("models")) {
                                db.createObjectStore("models");
                            }
                        };

                        request.onsuccess = (event) => {
                            try {
                                const db = event.target.result;

                                // Ensure object store exists
                                if (!db.objectStoreNames.contains("models")) {
                                    console.log("HB==Creating missing object store");
                                    db.close();
                                    // Reopen with higher version to trigger onupgradeneeded
                                    const reopenRequest = indexedDB.open("nsfw-models", 2);
                                    reopenRequest.onupgradeneeded = (evt) => {
                                        const newDb = evt.target.result;
                                        if (!newDb.objectStoreNames.contains("models")) {
                                            newDb.createObjectStore("models");
                                        }
                                    };
                                    reopenRequest.onsuccess = (evt) => {
                                        const newDb = evt.target.result;
                                        try {
                                            const transaction = newDb.transaction(["models"], "readwrite");
                                            const store = transaction.objectStore("models");
                                            store.put({ buffer: buffer }, modelName);
                                            transaction.oncomplete = () => {
                                                console.log("HB==Model cached successfully");
                                                newDb.close();
                                                resolve();
                                            };
                                            transaction.onerror = () => {
                                                console.error("HB==Transaction error:", transaction.error);
                                                newDb.close();
                                                resolve();
                                            };
                                        } catch (e) {
                                            console.error("HB==Cache write error:", e);
                                            newDb.close();
                                            resolve();
                                        }
                                    };
                                    reopenRequest.onerror = () => {
                                        console.error("HB==Failed to reopen database");
                                        resolve();
                                    };
                                    return;
                                }

                                const transaction = db.transaction(["models"], "readwrite");
                                const store = transaction.objectStore("models");
                                store.put({ buffer: buffer }, modelName);

                                transaction.oncomplete = () => {
                                    console.log("HB==Model cached successfully");
                                    db.close();
                                    resolve();
                                };
                                transaction.onerror = () => {
                                    console.error("HB==Transaction error:", transaction.error);
                                    db.close();
                                    resolve();
                                };
                            } catch (e) {
                                console.error("HB==Cache write error:", e);
                                resolve();
                            }
                        };
                        request.onerror = () => {
                            console.error("HB==Failed to open cache database:", request.error);
                            resolve();
                        };
                    })
                    .catch((err) => {
                        console.error("HB==Failed to fetch model:", err);
                        resolve();
                    });
            } catch (e) {
                console.error("HB==Caching error:", e);
                resolve();
            }
        });
    };

    nsfwModelSkip = async (input, config) => {
        let skipFrame = false;

        // Validation checks
        if (
            config.cacheSensitivity === 0 ||
            !input ||
            input.length === 0
        ) {
            return skipFrame;
        }

        try {
            if (!this.nsfwCache.lastInputTensor) {
                // First frame - just store it
                this.nsfwCache.lastInputTensor = new Float32Array(input);
            } else if (this.nsfwCache.lastInputTensor.length !== input.length) {
                // Input shape changed - reset cache
                this.nsfwCache.lastInputTensor = new Float32Array(input);
            } else {
                // Compare frames
                let diffSum = 0;

                for (let i = 0; i < input.length; i++) {
                    const diff = input[i] - this.nsfwCache.lastInputTensor[i];
                    diffSum += diff * diff;
                }

                const diffRelative = diffSum / input.length / 255 / 3;

                skipFrame = diffRelative <= (config.cacheSensitivity || 0);

                // Update cache
                this.nsfwCache.lastInputTensor = new Float32Array(input);
            }

            return skipFrame;
        } catch (error) {
            console.error("HB==Frame skip check failed:", error);
            return false;
        }
    };

    nsfwModelClassify = async (imageData, config = NSFW_CONFIG) => {
        if (!this._human) await this.initHuman();
        if (!this._nsfwModel) await this.initNsfwModel();

        try {
            // Check if we should skip this frame
            const skipAllowed = await this.nsfwModelSkip(imageData.data, config);
            const skipFrame = this.nsfwCache.skippedFrames < (config.skipFrames || 0);
            const skipTime =
                (config.skipTime || 0) >
                (performance?.now?.() || Date.now()) - this.nsfwCache.timestamp;

            // Run inference if skip conditions aren't met
            if (
                !skipAllowed ||
                !skipTime ||
                !skipFrame ||
                this.nsfwCache.predictions.length === 0
            ) {
                // Resize to model input size
                const resized = this._resizeImageCanvas(
                    imageData,
                    config.size,
                    config.size
                );

                // Normalize pixel values and convert to NCHW
                const normalized = this._normalizeImageData(resized.data, config.tfScalar);

                // Create ONNX tensor with batch dimension: [1, channels, height, width]
                const inputTensor = new ort.Tensor("float32", normalized, [
                    1,
                    3,
                    config.size,
                    config.size,
                ]);

                // Get input name from model
                const inputName = this._nsfwModel.inputNames[0];
                const feeds = { [inputName]: inputTensor };

                // Run inference
                const results = await this._nsfwModel.run(feeds);

                // Get output tensor
                const outputName = this._nsfwModel.outputNames[0];
                const outputTensor = results[outputName];
                const logits = outputTensor.data;

                // Process results
                this.nsfwCache.predictions = await this.getTopKClasses(
                    logits,
                    config.topK
                );
                this.nsfwCache.timestamp = performance?.now?.() || Date.now();
                this.nsfwCache.skippedFrames = 0;
            } else {
                this.nsfwCache.skippedFrames++;
            }

            return this.nsfwCache.predictions;
        } catch (error) {
            console.error("HB==NSFW Detection Error:", error);
            return [];
        }
    };

    // Helper: Resize ImageData using Canvas
    _resizeImageCanvas = (imageData, targetWidth, targetHeight) => {
        try {
            // Create source canvas with original image
            const srcCanvas = new OffscreenCanvas(imageData.width, imageData.height);
            const srcCtx = srcCanvas.getContext("2d");
            srcCtx.putImageData(imageData, 0, 0);

            // Create target canvas and resize
            const dstCanvas = new OffscreenCanvas(targetWidth, targetHeight);
            const dstCtx = dstCanvas.getContext("2d");

            // Use context.drawImage to resize
            dstCtx.drawImage(
                srcCanvas,
                0,
                0,
                imageData.width,
                imageData.height,
                0,
                0,
                targetWidth,
                targetHeight
            );

            return dstCtx.getImageData(0, 0, targetWidth, targetHeight);
        } catch (error) {
            console.error("HB==Canvas resize failed:", error);
            throw error;
        }
    };

    // Helper: Normalize ImageData to float32 and transpose to NCHW
    _normalizeImageData = (pixelData, divisor) => {
        const width = 224;
        const height = 224;
        const channels = 3;
        const normalized = new Float32Array(width * height * channels);

        // ImageNet mean and std
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        // Transpose from NHWC (R,G,B,R,G,B...) to NCHW (RRR...GGG...BBB...)
        for (let i = 0; i < width * height; i++) {
            // pixelData is RGBA, so we skip alpha (index 3)
            // Normalize to 0-1 then apply ImageNet normalization
            const r = (pixelData[i * 4] / divisor - mean[0]) / std[0];
            const g = (pixelData[i * 4 + 1] / divisor - mean[1]) / std[1];
            const b = (pixelData[i * 4 + 2] / divisor - mean[2]) / std[2];

            // NCHW layout
            normalized[i] = r;                     // R plane
            normalized[i + width * height] = g;    // G plane
            normalized[i + 2 * width * height] = b;// B plane
        }
        return normalized;
    };

    getTopKClasses = async (logits, topK) => {
        // logits is already a TypedArray from ONNX Runtime (no need to call .data())
        const values = logits instanceof Float32Array ? logits : new Float32Array(logits);

        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort((a, b) => {
            return b.value - a.value;
        });
        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }

        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
            topClassesAndProbs.push({
                className: getNsfwClasses()?.[topkIndices[i]].className,
                probability: topkValues[i],
                id: topkIndices[i],
            });
        }
        return topClassesAndProbs;
    };
}

const containsNsfw = (nsfwDetections, strictness) => {
    if (!nsfwDetections?.length) return false;
    let highestNsfwDelta = 0;
    let highestSfwDelta = 0;

    const nsfwClasses = getNsfwClasses(strictness);
    nsfwDetections.forEach((det) => {
        if (nsfwClasses?.[det.id].nsfw) {
            highestNsfwDelta = Math.max(
                highestNsfwDelta,
                det.probability - nsfwClasses[det.id].thresh
            );
        } else {
            highestSfwDelta = Math.max(
                highestSfwDelta,
                det.probability - nsfwClasses[det.id].thresh
            );
        }
    });
    return highestNsfwDelta > highestSfwDelta;
};

const genderPredicate = (gender, score, detectMale, detectFemale) => {
    return false;
};

const containsGenderFace = (detections, detectMale, detectFemale) => {
    if (!detections?.face?.length) {
        return false;
    }

    const faces = detections.face;
    if (detectMale || detectFemale)
        return faces.some(
            (face) =>
                face.age > 20 &&
                genderPredicate(
                    face.gender,
                    face.genderScore,
                    detectMale,
                    detectFemale
                )
        );
    else return false;
};
// export the human variable and the HUMAN_CONFIG object
export { getNsfwClasses, containsNsfw, containsGenderFace, Detector };
