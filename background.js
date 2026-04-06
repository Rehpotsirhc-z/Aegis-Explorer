// ============================================================
// Aegis Explorer - Background Service Worker
// Architecture: Hybrid image classification
//   1. NSFWJS (local, in-browser) — instant explicit content detection
//   2. YOLO (server) — all categories (drugs, gambling, games, profanity, explicit)
//   Server proxy for text (keeps OpenAI key secure)
// ============================================================

// Server URLs
const serverBase = "https://aegisexplorer.org/server";
//const serverBase = "http://localhost:8000";
const imageUrl = `${serverBase}/predict_image`;
const textUrl = `${serverBase}/predict_text`;

// URL-based cache: avoids re-processing the same image URL
const imageCache = new Map();
const MAX_CACHE_SIZE = 2000;

// Default confidence threshold
const DEFAULT_THRESHOLD = 0.5;

// File extensions to skip (icons, vectors — not meaningful for classification)
const SKIP_EXTENSIONS = /\.(svg|ico|gif|webp)(\?.*)?$/i;

// Category name → storage key mapping
const CATEGORIES = {
    profanity: "profanity",
    explicit: "explicit-content",
    drugs: "drugs",
    games: "web-based-games",
    gambling: "gambling",
    background: "background",
};

// ============================================================
// Utilities
// ============================================================

chrome.runtime.onInstalled.addListener(({ reason }) => {
    if (reason === "install") {
        chrome.tabs.create({ url: "barrier.html" });
    }
});

function recordCategory(category) {
    chrome.storage.local.get([`${category}-log`]).then((result) => {
        let currentTime = new Date().getTime();
        let log = Array.from(result[`${category}-log`] || []).filter(
            (time) => time > thirtyDaysAgo(),
        );
        chrome.storage.local.set({
            [`${category}-log`]: [...log, currentTime],
        });
    });
}

function thirtyDaysAgo() {
    let currentDate = new Date().getTime();
    let thirtyDays = 30 * 24 * 60 * 60 * 1000;
    return currentDate - thirtyDays;
}

function sendToTab(tabId, message) {
    if (tabId) {
        chrome.tabs.sendMessage(tabId, message).catch(() => {});
    } else {
        chrome.tabs.query({}, (tabs) => {
            tabs.forEach((tab) => {
                chrome.tabs.sendMessage(tab.id, message).catch(() => {});
            });
        });
    }
}

function cacheSet(key, value) {
    if (imageCache.size >= MAX_CACHE_SIZE) {
        const toDelete = Math.floor(MAX_CACHE_SIZE * 0.25);
        const keys = imageCache.keys();
        for (let i = 0; i < toDelete; i++) {
            imageCache.delete(keys.next().value);
        }
    }
    imageCache.set(key, value);
}

function chunk(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
}

// Online time tracking
setInterval(() => {
    chrome.storage.local.get(["onlineLog"]).then((result) => {
        log = Array.from(result.onlineLog || []);
        let time = new Date().getTime();
        log.push(time);
        chrome.storage.local.set({ onlineLog: log });
    });
}, 60000);

// ============================================================
// YOLO - Server-side classification (all categories)
// ============================================================

async function classifyImageWithYOLO(imgLink) {
    try {
        // Send the URL to the server — server downloads the image
        // (avoids CORS issues with cross-origin images)
        const response = await fetch(imageUrl, {
            method: "POST",
            body: JSON.stringify({ url: imgLink }),
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) return { error: `Server error: ${response.status}` };
        return await response.json();
    } catch (error) {
        console.error("YOLO classification failed:", error);
        return { error: error.message };
    }
}

function processYoloPredictions(predictions, confidenceThreshold) {
    if (!predictions || predictions.length === 0) {
        return { action: "revealImage", className: "background" };
    }

    let best = null;
    for (const pred of predictions) {
        if (pred.class === "background") continue;
        if (pred.confidence > confidenceThreshold) {
            if (!best || pred.confidence > best.confidence) {
                best = pred;
            }
        }
    }

    if (best) {
        return {
            action: "removeImage",
            className: best.class,
            confidence: best.confidence,
        };
    }

    return { action: "revealImage", className: "background" };
}

// ============================================================
// Text Classification via server proxy (keeps OpenAI key secure)
// ============================================================

async function classifyTextsViaServer(texts) {
    try {
        const response = await fetch(textUrl, {
            method: "POST",
            body: JSON.stringify({ texts }),
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
            console.error("Text API error:", response.status);
            return texts.map((t) => ({ text: t, flags: [] }));
        }

        return await response.json();
    } catch (error) {
        console.error("Text classification error:", error);
        return texts.map((t) => ({ text: t, flags: [] }));
    }
}

// ============================================================
// Main message handler
// ============================================================

chrome.runtime.onMessage.addListener(async (request, sender) => {
    const senderTabId = sender?.tab?.id;

    // --------------------------------------------------------
    // IMAGE PROCESSING (Hybrid: NSFWJS local + YOLO server)
    // Step 1: NSFWJS checks for explicit content locally (instant)
    //         If flagged → block immediately, skip YOLO
    // Step 2: YOLO checks all categories on server
    //         Catches drugs, gambling, games, profanity + second explicit check
    // --------------------------------------------------------
    if (request.images) {
        console.log(request.images.length, "images to process");
        const categoryCount = {};

        const categoryKeys = Object.values(CATEGORIES);
        const storageData = await chrome.storage.local.get([
            "confidence",
            ...categoryKeys,
        ]);
        const confidenceThreshold =
            typeof storageData.confidence === "number"
                ? storageData.confidence
                : DEFAULT_THRESHOLD;

        // Separate cached vs uncached
        const cachedImages = [];
        const uncachedImageLinks = [];

        for (const imageLink of request.images) {
            // Skip SVGs, ICOs, GIFs — not useful for content classification
            if (SKIP_EXTENSIONS.test(imageLink)) {
                sendToTab(senderTabId, { action: "revealImage", imageLink });
                continue;
            }

            if (imageCache.has(imageLink)) {
                cachedImages.push({
                    imageLink,
                    cached: imageCache.get(imageLink),
                });
            } else {
                uncachedImageLinks.push(imageLink);
            }
        }

        // Process cached results immediately
        for (const { imageLink, cached } of cachedImages) {
            sendToTab(senderTabId, { action: cached.action, imageLink });
            if (cached.action === "removeImage" && cached.category) {
                recordCategory(cached.category);
            }
        }

        if (uncachedImageLinks.length === 0) {
            console.log("All images served from cache");
            return;
        }

        console.log(
            `${cachedImages.length} cached, ${uncachedImageLinks.length} to classify`,
        );

        const classificationPromises = uncachedImageLinks.map(
            async (imageLink) => {
                try {
                    // --- YOLO server check (all categories) ---
                    const yoloResult =
                        await classifyImageWithYOLO(imageLink);

                    if (yoloResult?.error) {
                        // Server unavailable - reveal (NSFWJS already cleared explicit)
                        sendToTab(senderTabId, {
                            action: "revealImage",
                            imageLink,
                        });
                        cacheSet(imageLink, { action: "revealImage" });
                        return;
                    }

                    const decision = processYoloPredictions(
                        yoloResult.predictions,
                        confidenceThreshold,
                    );

                    if (decision.action === "removeImage") {
                        const storageKey = CATEGORIES[decision.className];
                        const isEnabled = storageData[storageKey] ?? true;

                        if (isEnabled) {
                            console.log(
                                `BLOCKED (YOLO): ${imageLink} | ${decision.className} (${(decision.confidence * 100).toFixed(1)}%)`,
                            );
                            recordCategory(storageKey || decision.className);
                            sendToTab(senderTabId, {
                                action: "removeImage",
                                imageLink,
                            });
                            cacheSet(imageLink, {
                                action: "removeImage",
                                category: storageKey || decision.className,
                                className: decision.className,
                            });
                        } else {
                            sendToTab(senderTabId, {
                                action: "revealImage",
                                imageLink,
                            });
                            cacheSet(imageLink, { action: "revealImage" });
                        }
                    } else {
                        sendToTab(senderTabId, {
                            action: "revealImage",
                            imageLink,
                        });
                        cacheSet(imageLink, { action: "revealImage" });
                    }

                    categoryCount[decision.className] =
                        (categoryCount[decision.className] || 0) + 1;
                } catch (error) {
                    console.error(`Error classifying ${imageLink}:`, error);
                    sendToTab(senderTabId, {
                        action: "revealImage",
                        imageLink,
                    });
                }
            },
        );

        // Fire and forget — don't block the message handler
        // Results are sent via sendToTab, no need to await
        Promise.all(classificationPromises).then(() => {
            console.log("Category counts:", categoryCount);
        });
    }

    // --------------------------------------------------------
    // TEXT PROCESSING (via OpenAI API - server proxy)
    // --------------------------------------------------------
    else if (request.texts) {
        console.log(request.texts.length, "texts to process");
        const categoryCount = {};

        const allSentences = request.texts.flatMap((rawText) => {
            return rawText
                .split(/(?<=[.!?])\s+/)
                .map((s) => s.trim())
                .filter((s) => s.length > 0);
        });

        if (allSentences.length === 0) return;

        const { confidence: storedConfidence } =
            await chrome.storage.local.get(["confidence"]);
        const confidenceThreshold = storedConfidence ?? 0.5;

        const categoryKeys = Object.values(CATEGORIES);
        const categoryToggles = await chrome.storage.local.get(categoryKeys);

        const BATCH_SIZE = 20;
        const batches = chunk(allSentences, BATCH_SIZE);

        const batchPromises = batches.map(async (batch) => {
            try {
                const textPredictions = await classifyTextsViaServer(batch);

                for (const result of textPredictions) {
                    const text = result.text;
                    const flags = Array.isArray(result.flags)
                        ? result.flags
                        : [];

                    const localCategoryCounter = {
                        profanity: 0,
                        explicit: 0,
                        drugs: 0,
                        games: 0,
                        gambling: 0,
                    };

                    flags.forEach((entry) => {
                        const cat = entry.category;
                        const conf = Number(entry.confidence) || 0;
                        if (
                            conf > confidenceThreshold &&
                            localCategoryCounter.hasOwnProperty(cat)
                        ) {
                            localCategoryCounter[cat] += 1;
                        }
                    });

                    let maxFlagged = null;
                    let maxCount = 0;
                    for (const [cat, cnt] of Object.entries(
                        localCategoryCounter,
                    )) {
                        if (cnt > maxCount) {
                            maxFlagged = cat;
                            maxCount = cnt;
                        }
                    }

                    if (!maxFlagged) {
                        categoryCount["background"] =
                            (categoryCount["background"] || 0) + 1;
                        continue;
                    }

                    const className = maxFlagged;
                    const storageKey = CATEGORIES[className];
                    const isEnabled = categoryToggles[storageKey] ?? true;

                    if (isEnabled) {
                        console.log(
                            `Text blocked: "${text}" | ${className}`,
                        );
                        sendToTab(senderTabId, {
                            action: "removeText",
                            text,
                        });
                        categoryCount[className] =
                            (categoryCount[className] || 0) + 1;
                    } else {
                        categoryCount["background"] =
                            (categoryCount["background"] || 0) + 1;
                    }
                }
            } catch (error) {
                console.error("Error processing text batch:", error);
            }
        });

        Promise.all(batchPromises).then(() => {
            console.log("Text category counts:", categoryCount);
        });
    }
});
