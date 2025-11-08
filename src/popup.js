var settings = {};

// Initialize popup when DOM is ready
document.addEventListener('DOMContentLoaded', initPopup);
window.addEventListener('load', initPopup);

// Also run immediately in case DOM is already loaded
initPopup();

function initPopup() {
    loadSettingsAndDisplay();
}

function loadSettingsAndDisplay() {
    chrome.storage.sync.get(["hb-settings"], (result) => {
        if (chrome.runtime.lastError) {
            console.error("Storage error:", chrome.runtime.lastError);
            return;
        }
        
        // Load settings from storage, or use empty object
        settings = result["hb-settings"] || {};
        
        // Ensure all default fields exist
        ensureDefaultSettings();
        
        // Display loaded settings
        displaySettings(settings);
        
        // Add event listeners
        addListeners();
        
        // Setup collapsible sections
        setupCollapsibles();
    });
}

function ensureDefaultSettings() {
    // Ensure all essential settings have default values
    if (typeof settings.status === 'undefined') settings.status = true;
    if (typeof settings.blurAmount === 'undefined') settings.blurAmount = 20;
    if (typeof settings.strictness === 'undefined') settings.strictness = 0.3;
    if (typeof settings.gray === 'undefined') settings.gray = true;
    if (typeof settings.blurImages === 'undefined') settings.blurImages = true;
    if (typeof settings.blurVideos === 'undefined') settings.blurVideos = true;
    if (typeof settings.blurryStartMode === 'undefined') settings.blurryStartMode = false;
    if (typeof settings.unblurImages === 'undefined') settings.unblurImages = false;
    if (typeof settings.unblurVideos === 'undefined') settings.unblurVideos = false;
    if (typeof settings.hideVideoToggle === 'undefined') settings.hideVideoToggle = false;
}

function displaySettings(settings) {
    setCheckbox("status", settings.status !== false);
    setRange("blurAmount", settings.blurAmount || 20);
    setRange("strictness", settings.strictness || 0.3);
    setCheckbox("gray", settings.gray !== false);
    setCheckbox("blurImages", settings.blurImages !== false);
    setCheckbox("blurVideos", settings.blurVideos !== false);
    setCheckbox("blurryStartMode", settings.blurryStartMode || false);
    setCheckbox("unblurImages", settings.unblurImages || false);
    setCheckbox("unblurVideos", settings.unblurVideos || false);
    setCheckbox("hideVideoToggle", settings.hideVideoToggle || false);
    updateSliderDisplay();
}

function setCheckbox(name, checked) {
    const el = document.querySelector(`input[name="${name}"]`);
    if (el) el.checked = checked;
}

function setRange(name, value) {
    const el = document.querySelector(`input[name="${name}"]`);
    if (el) el.value = value;
}

function updateSliderDisplay() {
    const blur = document.querySelector(`input[name="blurAmount"]`);
    if (blur) {
        document.getElementById("blur-value").textContent = blur.value + "%";
    }
    const strict = document.querySelector(`input[name="strictness"]`);
    if (strict) {
        document.getElementById("strictness-value").textContent = Math.round(strict.value * 100) + "%";
    }
}

function addListeners() {
    document.querySelectorAll("input[type='checkbox']").forEach(el => {
        el.addEventListener("change", () => {
            const key = el.name;
            const value = el.checked;
            
            // Update local settings object
            settings[key] = value;
            
            // Save to persistent storage
            saveSettings();
            
            // Update UI if needed
            updateSliderDisplay();
            
            // Send message to content script
            sendMsg(key);
        });
    });

    document.querySelectorAll("input[type='range']").forEach(el => {
        el.addEventListener("input", () => {
            const key = el.name;
            const value = key === "strictness" ? parseFloat(el.value) : parseInt(el.value);
            
            // Update local settings object
            settings[key] = value;
            
            // Update display immediately for better UX
            updateSliderDisplay();
            
            // Send message to content script
            sendMsg(key);
        });
        
        // Save on change event (when user releases slider)
        el.addEventListener("change", () => {
            saveSettings();
        });
    });
}

function saveSettings() {
    // Always save the complete settings object
    chrome.storage.sync.set({ "hb-settings": settings }, () => {
        if (chrome.runtime.lastError) {
            console.error("Error saving settings:", chrome.runtime.lastError);
        } else {
            console.log("Settings saved successfully");
        }
    });
}

function setupCollapsibles() {
    document.querySelectorAll(".card-title.collapsible").forEach(title => {
        title.addEventListener("click", function() {
            const id = this.getAttribute("data-toggle");
            const content = document.getElementById(id);
            const icon = this.querySelector(".collapse-icon");
            if (content) {
                content.classList.toggle("collapsed");
                icon.classList.toggle("open");
            }
        });
    });
}

function sendMsg(key) {
    try {
        chrome.runtime.sendMessage({
            type: "updateSettings",
            newSetting: {key: key, value: settings[key]}
        });
        chrome.tabs.query({currentWindow: true, active: true}, tabs => {
            if (tabs && tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: "updateSettings",
                    newSetting: {key: key, value: settings[key]}
                });
            }
        });
    } catch(e) {}
}
