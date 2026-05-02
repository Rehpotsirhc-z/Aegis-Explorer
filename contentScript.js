const seenImages = new Set();
const seenText = new Set();

// set keeping track of image URLs
const imageUrls = new Set();

const allRealSrcs = new Set();

// Minimum image dimension to process (skip icons, avatars, spacers)
const MIN_IMAGE_SIZE = 50;

// Per-URL safety reveal timer. Hides the image until either a verdict
// arrives (revealImage / removeImage clears the timer) or this timeout
// fires, in which case we fail-open and reveal.
const PENDING_REVEAL_MS = 30000;
const pendingTimers = new Map();

function startPendingTimer(url, img) {
    if (pendingTimers.has(url)) return;
    const id = setTimeout(() => {
        pendingTimers.delete(url);
        if (img && img.classList.contains("aegis-pending")) {
            img.classList.remove("aegis-pending");
            img.dataset.approved = "true";
        }
    }, PENDING_REVEAL_MS);
    pendingTimers.set(url, id);
}

function startPendingBgTimer(url, element) {
    if (pendingTimers.has(url)) return;
    const id = setTimeout(() => {
        pendingTimers.delete(url);
        if (element.dataset.originalBackgroundImage) {
            element.style.backgroundImage = `url(${url})`;
            element.dataset.approved = "true";
            element.removeAttribute("data-original-background-image");
        }
    }, PENDING_REVEAL_MS);
    pendingTimers.set(url, id);
}

function clearPendingTimer(url) {
    const id = pendingTimers.get(url);
    if (id !== undefined) {
        clearTimeout(id);
        pendingTimers.delete(url);
    }
}

// ============================================================
// NSFWJS local prefilter
// Block thresholds: Porn>0.6, Hentai>0.6, Sexy>0.8.
// On flag → block in-place and skip the YOLO server round-trip.
// On error (CORS-tainted canvas, model load failure) → fall through
//   to YOLO so we never silently let an image through.
// ============================================================
let nsfwModel = null;
let nsfwModelPromise = null;

function loadNsfwModel() {
    if (nsfwModel) return Promise.resolve(nsfwModel);
    if (nsfwModelPromise) return nsfwModelPromise;
    if (typeof nsfwjs === "undefined") {
        return Promise.reject(new Error("nsfwjs global not available"));
    }
    const modelUrl = chrome.runtime.getURL("models/nsfwjs/model.json");
    nsfwModelPromise = nsfwjs
        .load(modelUrl)
        .then((m) => {
            nsfwModel = m;
            return m;
        })
        .catch((err) => {
            nsfwModelPromise = null;
            throw err;
        });
    return nsfwModelPromise;
}

function waitForImageLoad(img) {
    if (img.complete && img.naturalWidth > 0) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            img.removeEventListener("load", onLoad);
            img.removeEventListener("error", onErr);
            clearTimeout(timer);
        };
        const onLoad = () => {
            cleanup();
            resolve();
        };
        const onErr = () => {
            cleanup();
            reject(new Error("image load error"));
        };
        const timer = setTimeout(() => {
            cleanup();
            reject(new Error("image load timeout"));
        }, 5000);
        img.addEventListener("load", onLoad, { once: true });
        img.addEventListener("error", onErr, { once: true });
    });
}

async function classifyImageLocally(img) {
    const model = await loadNsfwModel();
    await waitForImageLoad(img);
    const predictions = await model.classify(img);
    const scores = {};
    for (const p of predictions) scores[p.className] = p.probability;
    const blocked =
        (scores.Porn || 0) > 0.6 ||
        (scores.Hentai || 0) > 0.6 ||
        (scores.Sexy || 0) > 0.8;
    return { blocked, scores };
}

// Add a small text queue with debounce + max batch size
const textQueue = [];
let textFlushTimer = null;
const TEXT_BATCH_SIZE = 200; // max items per message to the background
const TEXT_FLUSH_DELAY = 150; // debounce flush delay (ms)

// Inject a style rule once for hiding pending images via CSS class
// This avoids blanking src (which breaks layout) and uses opacity + blur instead
const aegisStyle = document.createElement("style");
aegisStyle.textContent = `
    .aegis-pending {
        opacity: 0 !important;
        transition: opacity 0.15s ease-in;
    }
    .aegis-blocked {
        opacity: 0 !important;
    }
`;
(document.head || document.documentElement).appendChild(aegisStyle);

function flushTextQueue() {
    const batch = textQueue.splice(0, TEXT_BATCH_SIZE);
    if (batch.length) {
        try {
            chrome.runtime.sendMessage({ texts: batch });
        } catch (e) {
            console.error("Error sending text batch", e);
        }
    }
    if (textQueue.length) {
        textFlushTimer = setTimeout(flushTextQueue, TEXT_FLUSH_DELAY);
    } else {
        textFlushTimer = null;
    }
}

function queueTexts(texts) {
    const newOnes = [];
    texts.forEach((t) => {
        const s = (t || "").trim();
        if (s && !seenText.has(s)) {
            seenText.add(s);
            newOnes.push(s);
        }
    });
    if (newOnes.length) {
        textQueue.push(...newOnes);
        if (!textFlushTimer) {
            textFlushTimer = setTimeout(flushTextQueue, TEXT_FLUSH_DELAY);
        }
    }
}

// Check if an image is too small to be worth classifying
function isTooSmall(img) {
    // Use natural dimensions if available, fall back to rendered dimensions
    const w = img.naturalWidth || img.width || img.offsetWidth || 0;
    const h = img.naturalHeight || img.height || img.offsetHeight || 0;
    // If dimensions are 0, image hasn't loaded yet - don't skip it
    if (w === 0 && h === 0) return false;
    return w < MIN_IMAGE_SIZE || h < MIN_IMAGE_SIZE;
}

function extractImageLinks() {
    const images = document.querySelectorAll("img");
    // Pairs of { url, img } so callers (NSFWJS prefilter) can access the element.
    const imgEntries = [];

    Array.from(images)
        .filter((img) => img.dataset.approved !== "true")
        .forEach((img) => {
            const realSrc =
                img.dataset.src ||
                img.dataset.originalSrc ||
                img.dataset.lazySrc ||
                img.src;

            allRealSrcs.add(realSrc);

            if (!img.src.startsWith("data:") && isTooSmall(img)) {
                img.dataset.approved = "true";
                return;
            }

            if (img.src.startsWith("data:image/gif")) return;

            if (!img.dataset.originalSrc) {
                img.dataset.originalSrc = img.src;
            }
            img.dataset.originalAlt = img.alt;
            if (img.srcset !== "") {
                img.dataset.originalSrcset = img.srcset;
            }

            img.classList.add("aegis-pending");

            const url = img.dataset.originalSrc;
            if (url && !seenImages.has(url)) {
                seenImages.add(url);
                imgEntries.push({ url, img });
                startPendingTimer(url, img);
            }
        });

    const newImageLinks = imgEntries.map((e) => e.url);

    // Extract images from CSS background images - only check new elements
    // (querySelectorAll("*") with getComputedStyle is very expensive, so
    //  we limit to elements that are likely to have background images)
    const bgSelectors =
        "div, section, header, footer, article, aside, main, span, a, figure";
    const backgroundImages = Array.from(document.querySelectorAll(bgSelectors));

    backgroundImages.forEach((element) => {
        if (element.dataset.aegisBgChecked === "true") return;
        element.dataset.aegisBgChecked = "true";

        const backgroundImage =
            window.getComputedStyle(element).backgroundImage;
        if (backgroundImage && backgroundImage !== "none") {
            try {
                url = backgroundImage.match(/url\(["']?([^"']*)["']?\)/)[1];
                if (
                    element.dataset.approved !== "true" &&
                    !seenImages.has(url)
                ) {
                    seenImages.add(url);
                    newImageLinks.push(url);

                    element.dataset.originalBackgroundImage = url;
                    element.style.backgroundImage = "none";
                    startPendingBgTimer(url, element);
                }
            } catch (error) {
                // ignore invalid background-image values
            }
        }
    });

    return { imgEntries, bgLinks: newImageLinks.slice(imgEntries.length) };
}

function blockImageElement(img, url) {
    img.classList.remove("aegis-pending");
    img.classList.add("aegis-blocked");
    img.alt = "";
    img.removeAttribute("data-original-src");
    img.removeAttribute("data-original-alt");
    if (img.dataset.originalSrcset) {
        img.removeAttribute("data-original-srcset");
    }
    img.dataset.approved = "true";
    clearPendingTimer(url);
}

async function sendImages() {
    const { imgEntries, bgLinks } = extractImageLinks();
    const yoloLinks = bgLinks.slice();

    // NSFWJS prefilter: classify each <img> element. Blocked images skip
    // the YOLO server hop entirely; everything else is forwarded as before.
    await Promise.all(
        imgEntries.map(async ({ url, img }) => {
            try {
                const { blocked, scores } = await classifyImageLocally(img);
                if (blocked) {
                    console.log("BLOCKED (NSFWJS):", url, scores);
                    blockImageElement(img, url);
                    chrome.runtime
                        .sendMessage({ recordCategory: "explicit-content" })
                        .catch(() => {});
                    return;
                }
            } catch (e) {
                // Tainted canvas / load timeout / model failure → fall through
                // to YOLO so we never silently allow an unclassified image.
            }
            yoloLinks.push(url);
        }),
    );

    if (yoloLinks.length === 0) return;
    try {
        chrome.runtime.sendMessage({ images: yoloLinks });
    } catch (error) {
        console.error("Error sending images", error);
    }
}

function extractSentences() {
    sentences = [];

    const excludedTags = new Set([
        "script",
        "style",
        "noscript",
        "iframe",
        "code",
        "pre",
        "svg",
        "object",
        "embed",
    ]);

    function isVisible(node) {
        const style = window.getComputedStyle(node);
        return (
            style &&
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            parseFloat(style.opacity) > 0
        );
    }

    function extractTextFromNode(node) {
        if (!node) return;

        if (node.nodeType === Node.TEXT_NODE) {
            const parent = node.parentElement;
            if (
                parent &&
                isVisible(parent) &&
                !excludedTags.has(parent.tagName.toLowerCase())
            ) {
                const textContent = node.textContent.trim();
                if (textContent !== "") {
                    sentences.push(textContent);
                }
            }
        } else {
            node.childNodes.forEach((child) => extractTextFromNode(child));
        }
    }

    extractTextFromNode(document.body);

    // remove duplicates, empty strings, whitespace, too-short text, and seen text
    sentences = sentences
        .map((sentence) => sentence.trim())
        .filter((sentence) => sentence.length >= 4)
        .filter((sentence) => sentence !== "???")
        .filter((sentence) => sentence !== ":")
        .filter((sentence) => sentence !== "-")
        .filter((sentence) => !seenText.has(sentence));

    return sentences;
}

function sendText() {
    const textLinks = extractSentences();
    // enqueue instead of sending immediately
    queueTexts(textLinks);
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Set up a MutationObserver to detect changes in the DOM
// Optimized: only process added nodes instead of re-scanning entire DOM
const observer = new MutationObserver((mutations) => {
    let hasNewImages = false;
    let hasNewText = false;

    for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
            if (node.nodeType !== Node.ELEMENT_NODE) {
                if (
                    node.nodeType === Node.TEXT_NODE &&
                    node.textContent.trim()
                ) {
                    hasNewText = true;
                }
                continue;
            }
            // Check if the added node is or contains images
            if (node.tagName === "IMG" || node.querySelector?.("img")) {
                hasNewImages = true;
            }
            // Check for text content
            if (node.textContent?.trim()) {
                hasNewText = true;
            }
        }
    }

    if (hasNewImages) sendImages();
    if (hasNewText) sendText();
});

observer.observe(document, {
    childList: true,
    subtree: true,
});

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    if (message.action === "removeImage" && message.imageLink) {
        console.log("Removing image", message.imageLink);
        clearPendingTimer(message.imageLink);

        const images = Array.from(
            document.querySelectorAll("img[data-original-src]"),
        ).filter((img) => img.dataset.originalSrc === message.imageLink);
        images.forEach((image) => {
            // Use CSS class for blocking (keeps layout intact)
            image.classList.remove("aegis-pending");
            image.classList.add("aegis-blocked");
            image.alt = "";
            image.removeAttribute("data-original-src");
            image.removeAttribute("data-original-alt");
            if (image.dataset.originalSrcset) {
                image.removeAttribute("data-original-srcset");
            }
        });

        // Handle background images
        const elements = Array.from(
            document.querySelectorAll("[data-original-background-image]"),
        ).filter(
            (el) => el.dataset.originalBackgroundImage === message.imageLink,
        );
        elements.forEach((element) => {
            element.style.backgroundImage = "none";
            element.removeAttribute("data-original-background-image");
        });
    } else if (message.action === "revealImage" && message.imageLink) {
        console.log("Revealing image", message.imageLink);
        clearPendingTimer(message.imageLink);

        const images = Array.from(
            document.querySelectorAll("img[data-original-src]"),
        ).filter((img) => img.dataset.originalSrc === message.imageLink);
        images.forEach((image) => {
            // Remove hiding class and restore visibility
            image.classList.remove("aegis-pending");
            image.alt = image.dataset.originalAlt || "";
            if (image.dataset.originalSrcset) {
                image.srcset = image.dataset.originalSrcset;
                image.removeAttribute("data-original-srcset");
            }
            image.dataset.approved = "true";
            image.removeAttribute("data-original-src");
            image.removeAttribute("data-original-alt");
        });

        // Handle background images
        const elements = Array.from(
            document.querySelectorAll("[data-original-background-image]"),
        ).filter(
            (el) => el.dataset.originalBackgroundImage === message.imageLink,
        );
        elements.forEach((element) => {
            console.log("Revealing background image", message.imageLink);
            element.style.backgroundImage = `url(${element.dataset.originalBackgroundImage})`;
            element.dataset.approved = "true";
            element.removeAttribute("data-original-background-image");
        });
    } else if (message.action === "removeText" && message.text) {
        const text = message.text.trim();
        console.log("Removing: ", text);

        function removeTextFromNode(node) {
            if (node.nodeType === Node.TEXT_NODE) {
                const content = node.textContent.trim();
                if (content === "") return;

                // Only replace if the text node matches exactly or
                // the flagged text is a full word/sentence within the node
                if (content === text) {
                    // Exact match — replace entire node content
                    node.textContent = "???";
                } else if (
                    text.length >= 8 &&
                    node.textContent.includes(text)
                ) {
                    // Longer flagged text — safe to do substring replace
                    node.textContent = node.textContent.replace(
                        new RegExp(escapeRegExp(text), "gi"),
                        "???",
                    );
                }
            } else {
                node.childNodes.forEach((child) => removeTextFromNode(child));
            }
        }

        removeTextFromNode(document.body);
    }
});

function debounce(func, wait = 100) {
    let t;
    return () => {
        clearTimeout(t);
        t = setTimeout(func, wait);
    };
}

// Initial scan: run when DOM is ready and again when page fully loads
// (MutationObserver may miss content rendered during initial page parse)
document.addEventListener("DOMContentLoaded", () => {
    sendImages();
    sendText();
});
window.addEventListener("load", () => {
    sendImages();
    sendText();
});

(function () {
    // Reduced debounce from 1000ms to 150ms for faster detection
    const lazySend = debounce(() => sendImages(), 150);

    const srcDesc = Object.getOwnPropertyDescriptor(
        HTMLImageElement.prototype,
        "src",
    );

    Object.defineProperty(HTMLImageElement.prototype, "src", {
        ...srcDesc,
        set(value) {
            srcDesc.set.call(this, value);

            if (this.dataset.approved === "true") return;

            if (value !== "") {
                lazySend();
            }
        },
    });
})();
