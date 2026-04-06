const seenImages = new Set();
const seenText = new Set();

// set keeping track of image URLs
const imageUrls = new Set();

const allRealSrcs = new Set();

// Minimum image dimension to process (skip icons, avatars, spacers)
const MIN_IMAGE_SIZE = 50;

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

    const newImageLinks = Array.from(images)
        .filter((img) => img.dataset.approved !== "true")
        .map((img) => {
            const realSrc =
                img.dataset.src ||
                img.dataset.originalSrc ||
                img.dataset.lazySrc ||
                img.src;

            allRealSrcs.add(realSrc);

            // Skip tiny images (icons, avatars, spacers)
            if (isTooSmall(img)) {
                img.dataset.approved = "true";
                return "";
            }

            if (!img.src.startsWith("data:image/gif")) {
                if (!img.dataset.originalSrc) {
                    // Only set the originalSrc once, when it has the correct value
                    img.dataset.originalSrc = img.src;
                }
                img.dataset.originalAlt = img.alt;
                if (img.srcset !== "") {
                    img.dataset.originalSrcset = img.srcset;
                }

                // Use CSS class to hide instead of blanking src
                // This preserves layout and prevents broken image indicators
                img.classList.add("aegis-pending");

                // Safety timeout: auto-reveal if classification doesn't respond
                setTimeout(() => {
                    if (img.classList.contains("aegis-pending")) {
                        img.classList.remove("aegis-pending");
                        img.dataset.approved = "true";
                    }
                }, 5000);

                return img.dataset.originalSrc;
            } else {
                return "";
            }
        })
        .filter((src) => src !== "" && !seenImages.has(src));

    newImageLinks.forEach((src) => seenImages.add(src));

    // Extract images from CSS background images - only check new elements
    // (querySelectorAll("*") with getComputedStyle is very expensive, so
    //  we limit to elements that are likely to have background images)
    const bgSelectors = "div, section, header, footer, article, aside, main, span, a, figure";
    const backgroundImages = Array.from(document.querySelectorAll(bgSelectors));

    backgroundImages.forEach((element) => {
        if (element.dataset.aegisBgChecked === "true") return;
        element.dataset.aegisBgChecked = "true";

        const backgroundImage =
            window.getComputedStyle(element).backgroundImage;
        if (backgroundImage && backgroundImage !== "none") {
            try {
                url = backgroundImage.match(/url\(["']?([^"']*)["']?\)/)[1];
                if (element.dataset.approved !== "true" && !seenImages.has(url)) {
                    seenImages.add(url);
                    newImageLinks.push(url);

                    element.dataset.originalBackgroundImage = url;
                    element.style.backgroundImage = "none";
                }
            } catch (error) {
                // ignore invalid background-image values
            }
        }
    });

    return newImageLinks;
}

function sendImages() {
    const imageLinks = extractImageLinks();
    try {
        if (imageLinks.length > 0) {
            chrome.runtime.sendMessage({ images: imageLinks });
        }
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
                if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
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

        const images = document.querySelectorAll(
            `img[data-original-src="${message.imageLink}"]`,
        );
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
        const elements = document.querySelectorAll(
            `*[data-original-background-image="${message.imageLink}"]`,
        );
        elements.forEach((element) => {
            element.style.backgroundImage = "none";
            element.removeAttribute("data-original-background-image");
        });
    } else if (message.action === "revealImage" && message.imageLink) {
        console.log("Revealing image", message.imageLink);

        const images = document.querySelectorAll(
            `img[data-original-src="${message.imageLink}"]`,
        );
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
        const elements = document.querySelectorAll(
            `*[data-original-background-image="${message.imageLink}"]`,
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
                } else if (text.length >= 8 && node.textContent.includes(text)) {
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
