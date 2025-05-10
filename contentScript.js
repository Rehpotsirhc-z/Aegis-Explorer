const seenImages = new Set();
const seenText = new Set();

// set keeping track of image URLs
const imageUrls = new Set();

const allRealSrcs = new Set();

function extractImageLinks() {
    const images = document.querySelectorAll("img");

    const newImageLinks = Array.from(images)
        .filter((img) => img.dataset.approved !== "true")
        .map((img) => {

            const realSrc = img.dataset.src
                || img.dataset.originalSrc
                || img.dataset.lazySrc
                || img.dataset.originalSrc
                || img.src;

            allRealSrcs.add(realSrc);

            if (!img.src.startsWith("data:image/gif")) {
                if (!img.dataset.originalSrc) {
                    // Only set the originalSrc once, when it has the correct value
                    // The browser resets the src to the URL of the webpage, so
                    // what happens is that it hasn't been added to the seenImages,
                    // and so when this function is rerun, it resets it with the
                    // wrong value.
                    img.dataset.originalSrc = img.src;
                }
                img.dataset.originalAlt = img.alt;
                if (img.srcset !== "") {
                    img.dataset.originalSrcset = img.srcset;
                    img.srcset = "";
                }
                img.src = "";
                img.alt = "";

                return img.dataset.originalSrc;
            } else {
                return "";
            }
        })
        .filter((src) => src !== "" && !seenImages.has(src)); // We do this after so that they still disappear if not approved

    newImageLinks.forEach((src) => seenImages.add(src));

    // Extract images from CSS background images
    const backgroundImages = Array.from(document.querySelectorAll("*"));

    backgroundImages.forEach((element) => {
        const backgroundImage =
            window.getComputedStyle(element).backgroundImage;
        if (backgroundImage && backgroundImage !== "none") {
            try {
                url = backgroundImage.match(/url\(["']?([^"']*)["']?\)/)[1];
                if (element.dataset.approved !== "true") {
                    console.log("Background image found:", url);
                    // seenImages.add(url);
                    // if (!seenImages.has(url)) {
                    newImageLinks.push(url);
                    // }

                    element.dataset.originalBackgroundImage = url;
                    element.style.backgroundImage = "none";
                }
            } catch (error) {
                // console.error("Error extracting background image", error);
            }
        }
    });

    console.log(`${newImageLinks.length} new images`);
    return newImageLinks;
}

function sendImages() {
    const imageLinks = extractImageLinks();
    console.log(allRealSrcs)
    try {
        if (imageLinks.length > 0) {
            chrome.runtime.sendMessage({ images: imageLinks });
        }
    } catch (error) {
        console.error("Error sending images", error);
    }
}

function extractSentences() {
    // const textContent = document.body.innerText;
    // const sentences = textContent.match(/[^.!?]*[.!?]/g) || [];
    // return sentences;
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
        if (node.nodeType === Node.TEXT_NODE) {
            const parent = node.parentElement;
            if (
                parent &&
                isVisible(parent) &&
                !excludedTags.has(parent.tagName.toLowerCase())
            ) {
                textContent = node.textContent.trim();
                if (textContent !== "") {
                    sentences.push(textContent);
                }
            }
        } else {
            node.childNodes.forEach((child) => extractTextFromNode(child));
        }
    }

    extractTextFromNode(document.body);

    // remove duplicates, empty strings, whitespace, and seen text
    sentences = sentences
        .map((sentence) => sentence.trim())
        .filter((sentence) => sentence !== "")
        .filter((sentence) => sentence !== "???")
        .filter((sentence) => sentence !== ":")
        .filter((sentence) => sentence !== "-")
        // .filter((sentence) => !sentence.includes("???"))
        // .filter((sentence) => !sentence.includes(":"))
        .filter((sentence) => !seenText.has(sentence));

    return sentences;
}

function sendText() {
    const textLinks = extractSentences();
    textLinks.forEach((text) => seenText.add(text.trim()));

    console.log("textLinks", textLinks);
    console.log("seenText", seenText);

    try {
        if (textLinks.length > 0) {
            chrome.runtime.sendMessage({ texts: textLinks });
        }
    } catch (error) {
        console.error("Error sending text", error);
    }
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}

// Set up a MutationObserver to detect change in the DOM
const observer = new MutationObserver(() => {
    sendImages();
    sendText();
});

observer.observe(document, {
    childList: true,
    subtree: true,
    //     // attributes: true,
    //     // attributesFilter: ["src"],
});

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    if (message.action === "removeImage" && message.imageLink) {
        console.log("Removing image", message.imageLink);

        const images = document.querySelectorAll(
            `img[data-original-src="${message.imageLink}"]`,
        );
        images.forEach((image) => {
            image.src = "";
            image.alt = "";
            if (image.srcset === "" && image.dataset.originalSrcset) {
                image.srcset = "";
                image.removeAttribute("data-original-srcset");
            }
            image.removeAttribute("data-original-src");
            image.removeAttribute("data-original-alt");
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
            `img[src=""][data-original-src="${message.imageLink}"]`,
        );
        images.forEach((image) => {
            image.src = image.dataset.originalSrc;
            image.alt = image.dataset.originalAlt;
            if (image.srcset === "" && image.dataset.originalSrcset) {
                image.srcset = image.dataset.originalSrcset;
                image.removeAttribute("data-original-srcset");
            }
            image.dataset.approved = "true";
            image.style.display = "block";
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
            element.style.display = "block";
            element.removeAttribute("data-original-background-image");
        });
    } else if (message.action === "removeText" && message.text) {
        // remove all instances of the text in the document
        const text = message.text.trim();
        console.log("Removing: ", text);

        function removeTextFromNode(node) {
            if (node.nodeType === Node.TEXT_NODE) {
                textContent = node.textContent;
                if (textContent.trim() === "") {
                    return;
                }
                // console.log("Target:", text);
                // console.log(node.textContent);
                // node.textContent = node.textContent.replace(text, "???");
                if (node.textContent.includes(text)) {
                    // console.log("Result: ", node.textContent);
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

// window.addEventListener("load", () => {
//     sendImages();
//     sendText();
// });

// document.addEventListener("DOMContentLoaded", () => {
//     sendImages();
//     sendText();
// });

// sendImages();


function debounce(func, wait = 100) {
    let t;
    return () => {
        clearTimeout(t);
        t = setTimeout(func, wait);
    };
}

;(function() {
    const lazySend = debounce(() => sendImages(), 1000);

    const srcDesc = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');

    const seenElements = new WeakSet();

    // function resetSeenElements() {
    //     seenElements.clear();
    // }

    Object.defineProperty(HTMLImageElement.prototype, 'src', {
        ...srcDesc,
        set(value) {
            console.log('[IMG LOG] Setting src:', value, ' on ', this);
            srcDesc.set.call(this, value);
            // sendImages();
            // if (this.dataset.approved !== 'true' && !seenElements.has(this) && value !== '') {
                // seenElements.add(this);
                // lazySend();
            // }

            if (this.dataset.approved === 'true') return;

            // if (seenElements.has(this)) return;

            // seenElements.add(this);
            if (value !== '') {
                lazySend();
            }
        }
    });
})();