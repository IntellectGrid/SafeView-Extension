// when installed or updated load settings

const defaultSettings = {
    status: true,
    blurryStartMode: false,
    blurAmount: 20,
    blurImages: true,
    blurVideos: true,
    blurMale: false,
    blurFemale: true,
    unblurImages: false,
    unblurVideos: false,
    gray: true,
    strictness: 0.2, // goes from 0 to 1,
    blurryStartTimeout: 7000, // milliseconds (7 seconds default)
    whitelist: [],
};

chrome.runtime.onInstalled.addListener(function (details) {
    chrome.storage.sync.get(["hb-settings"], function (result) {
        if (
            result["hb-settings"] === undefined ||
            result["hb-settings"] === null
        ) {
            chrome.storage.sync.set({ "hb-settings": defaultSettings });
        } else {
            // if there are any new settings, add them to the settings object
            chrome.storage.sync.set({
                "hb-settings": { ...defaultSettings, ...result["hb-settings"] },
            });
        }
    });

    // context menu: "enable detection on this video"
    chrome.contextMenus.create({
        id: "enable-detection",
        title: "Enable for this video",
        contexts: ["all"],
        type: "checkbox",
        enabled: true,
        checked: true,
    });

    if (details?.reason === "install") {
        chrome.tabs.create({
            url: chrome.runtime.getURL("src/install.html"),
        });
    } else if (details?.reason === "update") {
    }
});

const createOffscreenDoc = async () => {
    if (await chrome.offscreen.hasDocument()) {
        console.log("offscreen document already exists");
        return;
    }
    chrome?.offscreen
        .createDocument({
            url: chrome.runtime.getURL("src/offscreen.html"),
            reasons: ["DOM_PARSER"],
            justification: "Process Images",
        })
        .then((document) => {
            console.log("offscreen document created");
        })
        .finally(() => { });
};

createOffscreenDoc();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // Guard against undefined request
    if (!request) {
        return false;
    }

    if (request.type === "getSettings") {
        chrome.storage.sync.get(["hb-settings"], function (result) {
            sendResponse(result["hb-settings"]);

            const isVideoEnabled =
                result["hb-settings"].status &&
                result["hb-settings"].blurVideos;
            chrome.contextMenus.update("enable-detection", {
                enabled: isVideoEnabled,
                checked: isVideoEnabled,
                title: isVideoEnabled
                    ? "Enabled for this video"
                    : "Please enable video detection in settings",
            });
        });
        return true; // Will respond asynchronously
    } else if (request.type === "video-status") {
        chrome.contextMenus.update("enable-detection", {
            checked: request.status,
        });
        return true;
    } else if (request.type === "reloadExtension") {
        // kill the offscreen document
        chrome?.offscreen?.closeDocument();
        // recreate the offscreen document
        createOffscreenDoc();
        return false; // Synchronous, no response needed
    } else if (request.type === "updateSettings") {
        // react to live setting changes for context menu visibility
        const { key, value } = request.newSetting || {};
        if (key === "hideVideoToggle") {
            // hide or show the context menu
            chrome.contextMenus.update("enable-detection", {
                enabled: !value,
                title: value
                    ? "Video detection toggle hidden"
                    : "Enable for this video",
            });
        }
        if (key === "status" || key === "blurVideos") {
            // recompute enable state only if not hidden
            chrome.storage.sync.get(["hb-settings"], (result) => {
                const s = result["hb-settings"] || {};
                if (s.hideVideoToggle) return; // keep hidden state
                const isVideoEnabled = s.status && s.blurVideos;
                chrome.contextMenus.update("enable-detection", {
                    enabled: isVideoEnabled,
                    checked: isVideoEnabled,
                    title: isVideoEnabled
                        ? "Enabled for this video"
                        : "Please enable video detection in settings",
                });
            });
        }
        return false; // Synchronous, no response needed
    }

    return false; // Default: no response for unknown message types
});



chrome.contextMenus.onClicked.addListener((info, tab) => {
    console.log("HB== context menu clicked", info, tab);
    if (info.menuItemId === "enable-detection") {
        if (info.checked) {
            chrome.tabs.sendMessage(tab.id, {
                type: "enable-detection",
            });
        } else {
            chrome.tabs.sendMessage(tab.id, {
                type: "disable-detection",
            });
        }
    }

    return true;
});

// on uninstall
chrome.runtime.setUninstallURL("https://docs.google.com/forms/d/e/1FAIpQLSdk2xdPgHA9giM3yaRKlhB3xOJe8pRHERVPJ1Mf8iMWEnsTvg/viewform?usp=header");
