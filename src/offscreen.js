import {
    containsNsfw,
    containsGenderFace,
    Detector,
} from "./modules/detector.js";
import Queue from "./modules/queues.js";
import Settings from "./modules/settings.js";

var settings;
var queue;
var detector = new Detector();

const loadModels = async () => {
    try {
        // Configure ONNX Runtime to look for WASM/MJS files in assets/
        if (typeof ort !== 'undefined') {
            ort.env.wasm.wasmPaths = chrome.runtime.getURL("src/assets/");
        }

        await detector.initHuman();
        await detector.initNsfwModel();
        detector.human.events?.addEventListener("error", (e) => {
            chrome.runtime.sendMessage({ type: "reloadExtension" });
        });
    } catch (e) {
        console.log("Error loading models", e);
    }
};

const handleImageDetection = (request, sender, sendResponse) => {
    queue.add(
        request.image,
        (result) => {
            sendResponse(result);
        },
        (error) => {
            error.type = "error";
            sendResponse(error);
        }
    );
};
let activeFrame = false;
let frameImage = new Image();

const handleVideoDetection = async (request, sender, sendResponse) => {
    const { frame } = request;
    const { data, timestamp, width, height } = frame;

    // Check if queue is busy. If so, skip this frame to prevent backlog and concurrency issues.
    // We check both loading and detection queues, and active processing.
    if (
        queue.loadingQueue.length > 0 ||
        queue.detectionQueue.length > 0 ||
        queue.activeProcessing > 0
    ) {
        sendResponse({ result: "skipped" });
        return;
    }

    // Create a mock image object for the queue (which expects an object with a src property)
    const imageRequest = {
        src: data,
        width: width || 0, // Optional, loadImage might handle it
        height: height || 0
    };

    queue.add(
        imageRequest,
        (result) => {
            sendResponse({ type: "detectionResult", result, timestamp });
        },
        (error) => {
            console.log("HB== error in video detection", error);
            sendResponse({ result: "error", error });
        }
    );
};

const startListening = () => {
    settings.listenForChanges();
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.type === "imageDetection") {
            handleImageDetection(request, sender, sendResponse);
        }
        if (request.type === "videoDetection") {
            handleVideoDetection(request, sender, sendResponse);
        }
        return true;
    });
};

const runDetection = async (img, isVideo = false) => {
    if (!settings?.shouldDetect() || !img) return false;

    // Convert image to ImageData using Canvas API
    const canvas = new OffscreenCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    // Convert ImageData to tensor for Human library (face detection)
    // This allocates GPU memory, so we MUST dispose it
    const tensor = detector.human.tf.browser.fromPixels(imageData);

    try {
        const nsfwResult = await detector.nsfwModelClassify(imageData);
        // console.log("offscreen nsfw result", nsfwResult);
        const strictness = settings.getStrictness() * (isVideo ? 0.75 : 1); // makes detection less strict for videos (to reduce false positives)
        activeFrame = false;
        if (containsNsfw(nsfwResult, strictness)) {
            return "nsfw";
        }
        if (!settings.shouldDetectGender()) {
            return false; // no need to run gender detection if it's not enabled
        }
        const predictions = await detector.humanModelClassify(tensor);
        // console.log("offscreen human result", predictions);

        if (
            containsGenderFace(
                predictions,
                settings.shouldDetectMale(),
                settings.shouldDetectFemale()
            )
        )
            return "face";
        return false;
    } finally {
        // Ensure tensor is disposed even if detection fails
        if (tensor) {
            detector.human.tf.dispose(tensor);
        }
    }
};

const init = async () => {
    let _settings = await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: "getSettings" }, (settings) => {
            resolve(settings);
        });
    });
    settings = await Settings.init(_settings["hb-settings"]);
    console.log("Settings loaded", settings);
    try {
        await loadModels();
        console.log("Models loaded", detector.human, detector.nsfwModel);
    } catch (error) {
        console.log("Error loading models", error);
        chrome.runtime.sendMessage({ type: "reloadExtension" });
        return;
    }

    queue = new Queue(runDetection);
    startListening();
};

init();
