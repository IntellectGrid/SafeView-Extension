import {
    containsNsfw,
    containsGenderFace,
    Detector,
} from "./modules/detector.js";
import OnnxDetector from "./modules/onnx_detector.js";
import Queue from "./modules/queues.js";
import Settings from "./modules/settings.js";

var settings;
var queue;
var detector = new Detector();
var onnxDetector = new OnnxDetector();
var currentRuntime = 'tfjs'; // track which runtime is active

const loadModels = async () => {
    try {
        // Determine which runtime to use
        currentRuntime = settings?.getSettings()?.selectedModel || 'tfjs';
        
        if (currentRuntime === 'onnx_b0') {
            console.log('Loading ONNX EfficientNet-B0 model...');
            await onnxDetector.init();
            // Still load human model for gender detection (if needed)
            await detector.initHuman();
        } else {
            // Default TFJS runtime
            console.log('Loading TFJS models (human + NSFW)...');
            await detector.initHuman();
            await detector.initNsfwModel();
        }
        
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
    const { data, timestamp } = frame;
    if (activeFrame) {
        sendResponse({ result: "skipped" });
        return;
    }
    activeFrame = true;
    frameImage.onload = () => {
        runDetection(frameImage, true)
            .then((result) => {
                activeFrame = false;
                sendResponse({ type: "detectionResult", result, timestamp });
            })
            .catch((e) => {
                console.log("HB== error in detectImage", e);
                activeFrame = false;
                sendResponse({ result: "error" });
            });
    };
    frameImage.onerror = (e) => {
        console.log("HB== image error", e);
        activeFrame = false;
        sendResponse({ result: "error" });
    };
    frameImage.src = data;
};

const startListening = () => {
    settings.listenForChanges();
    
    // Listen for model/runtime changes
    document.addEventListener('changeModel', async (event) => {
        console.log('Model changed, reloading models...');
        await loadModels();
    });
    
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
    
    const strictness = settings.getStrictness() * (isVideo ? 0.75 : 1); // less strict for videos
    activeFrame = false;
    
    // Choose runtime based on current selection
    currentRuntime = settings?.getSettings()?.selectedModel || 'tfjs';
    
    let nsfwResult;
    if (currentRuntime === 'onnx_b0') {
        // ONNX runtime: img is already an Image or canvas
        nsfwResult = await onnxDetector.nsfwModelClassify(img);
    } else {
        // TFJS runtime: use existing flow
        const tensor = detector.human.tf.browser.fromPixels(img);
        nsfwResult = await detector.nsfwModelClassify(tensor);
        
        if (containsNsfw(nsfwResult, strictness)) {
            detector.human.tf.dispose(tensor);
            return "nsfw";
        }
        if (!settings.shouldDetectGender()) {
            detector.human.tf.dispose(tensor);
            return false;
        }
        const predictions = await detector.humanModelClassify(tensor);
        detector.human.tf.dispose(tensor);
        if (
            containsGenderFace(
                predictions,
                settings.shouldDetectMale(),
                settings.shouldDetectFemale()
            )
        )
            return "face";
        return false;
    }
    
    // For ONNX: only NSFW detection is supported (no gender detection yet)
    // containsNsfw expects array of {className, probability}
    if (containsNsfw(nsfwResult, strictness)) {
        return "nsfw";
    }
    return false;
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
