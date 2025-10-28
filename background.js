// baseUrl = "http://14.187.67.90:10984";
baseUrl = "http://localhost:5000";
imageUrl = `${baseUrl}/predict_image`;
textUrl = `${baseUrl}/predict_text`;
suppTextUrl = `${baseUrl}/predict_text_supplementary`;

// // set keeping track of image URLs
// const imageUrls = new Set();

chrome.runtime.onInstalled.addListener(({ reason }) => {
    if (reason === "install") {
        chrome.tabs.create({
            url: "barrier.html",
        });
    }
});

function dataUrlToBlob(dataUrl) {
    const [header, data] = dataUrl.split(",");
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(data);
    const array = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
        array[i] = binary.charCodeAt(i);
    }

    return new Blob([array], { type: mime });
}

async function downloadImage(url) {
    if (!url) return;
    let blob;

    if (url.startsWith("data:")) {
        blob = dataUrlToBlob(url);
    } else {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.log(`Failed to fetch image from URL: "${url}"`);
                return null;
            }
            blob = await response.blob();
        } catch (error) {
            console.error(`Error fetching image from URL (${url}):`, error);
            return null;
        }
    }

    if (!blob.type.startsWith("image/")) {
        console.log(`Skipping non-image URL: "${url}" | Type: "${blob.type}"`);
        return null;
    }

    if (blob.type.startsWith("image/svg")) {
        console.log(`Skipping SVG image from URL: "${url}"`);
        return null;
    }

    try {
        // Create an ImageBitmap to access image dimensions
        const img = await createImageBitmap(blob);

        // TODO
        // if (img.width < 32 || img.height < 32) {
        //     // console.log(
        //     //     `Skipping image from URL: "${url}" | Dimensions: ${img.width}x${img.height}`,
        //     // );
        //     return null;
        // }

        // // Create an offscreen canvas to process the image
        // const canvas = new OffscreenCanvas(img.width, img.height);
        // const ctx = canvas.getContext("2d");
        // ctx.drawImage(img, 0, 0);

        // // Convert canvas content to JPEG
        // const jpgBlob = await canvas.convertToBlob({
        //     type: "image/jpeg",
        //     quality: 1,
        // });

        // const filename = url.split("/").pop().split(".").shift() + ".jpg";

        return new File([blob], "image", { type: blob.type });
        // return new File([blob], "image", { type: "image" });
    } catch (error) {
        console.error(`Error processing image from URL (${url}):`, error);
        return null;
    }
}

function recordCategory(category) {
    chrome.storage.local.get([`${category}-log`]).then((result) => {
        let currentTime = new Date().getTime();
        let log = Array.from(result[`${category}-log`] || []).filter(
            (time) => time > thirtyDaysAgo(),
        );
        console.log("Log: ", result);
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

setInterval(() => {
    chrome.storage.local.get(["onlineLog"]).then((result) => {
        log = Array.from(result.onlineLog || []);
        console.log(log);

        let time = new Date().getTime();
        log.push(time);
        chrome.storage.local.set({ onlineLog: log });
    });
}, 60000);

chrome.runtime.onMessage.addListener(async (request) => {
    if (request.images) {
        console.log(request.images.length, "images to process");
        const categoryCount = {};

        // Download all images concurrently and keep track of URLs
        const imagePromises = request.images.map(async (imageLink) => {
            const image = await downloadImage(imageLink);
            return { image, imageLink };
        });

        const imagesWithUrls = (await Promise.all(imagePromises)).filter(
            ({ image }) => image,
        );

        console.log(imagesWithUrls);

        console.log(imagesWithUrls.length, "images downloaded");

        // Create and send requests for all images concurrently
        const predictionPromises = imagesWithUrls.map(
            async ({ image, imageLink }) => {
                try {
                    const formData = new FormData();
                    formData.append("image", image);

                    const response = await fetch(imageUrl, {
                        method: "POST",
                        body: formData,
                    });

                    const { predictions: [prediction] = [] } =
                        await response.json();

                    chrome.storage.local.get(["confidence"]).then((result) => {
                        confidenceThreshold = result.confidence || 0.5;

                        if (prediction) {
                            const { class: className, confidence } = prediction;
                            if (confidence > confidenceThreshold) {
                                if (className !== "background") {
                                    console.log(
                                        `URL: ${imageLink} | Prediction: ${className} (${(confidence * 100).toFixed(2)}%)`,
                                    );

                                    const categories = {
                                        profanity: "profanity",
                                        // social: "social-media-and-forums",
                                        // monetary: "monetary-transactions",
                                        explicit: "explicit-content",
                                        drugs: "drugs",
                                        games: "web-based-games",
                                        gambling: "gambling",
                                        background: "background",
                                    };

                                    Object.entries(categories).forEach(
                                        ([key, value]) => {
                                            chrome.storage.local
                                                .get([value])
                                                .then((result) => {
                                                    if (
                                                        className === key &&
                                                        (result[value] ||
                                                            result[value] ===
                                                                undefined)
                                                    ) {
                                                        console.log(
                                                            "Category: ",
                                                            value,
                                                        );

                                                        recordCategory(value);

                                                        chrome.tabs.query(
                                                            {},
                                                            (tabs) => {
                                                                tabs.forEach(
                                                                    (tab) => {
                                                                        chrome.tabs
                                                                            .sendMessage(
                                                                                tab.id,
                                                                                {
                                                                                    action: "removeImage",
                                                                                    imageLink,
                                                                                },
                                                                            )
                                                                            .catch(
                                                                                (
                                                                                    error,
                                                                                ) => {
                                                                                    // console.error(
                                                                                    //     `Error removing image from URL (${imageLink}):`,
                                                                                    //     error,
                                                                                    // );
                                                                                },
                                                                            );
                                                                    },
                                                                );
                                                            },
                                                        );
                                                        // chrome.runtime.sendMessage({ action: "removeImage", imageLink: imageLink });
                                                    } else if (
                                                        className === key &&
                                                        !(
                                                            result[value] ||
                                                            result[value] ===
                                                                undefined
                                                        )
                                                    ) {
                                                        recordCategory(
                                                            "background",
                                                        );

                                                        chrome.tabs.query(
                                                            {},
                                                            (tabs) => {
                                                                tabs.forEach(
                                                                    (tab) => {
                                                                        chrome.tabs
                                                                            .sendMessage(
                                                                                tab.id,
                                                                                {
                                                                                    action: "revealImage",
                                                                                    imageLink,
                                                                                },
                                                                            )
                                                                            .catch(
                                                                                (
                                                                                    error,
                                                                                ) => {
                                                                                    // console.error(
                                                                                    //     `Error revealing image from URL (${imageLink}):`,
                                                                                    //     error,
                                                                                    // );
                                                                                },
                                                                            );
                                                                    },
                                                                );
                                                            },
                                                        );
                                                        // chrome.runtime.sendMessage({ action: "revealImage", imageLink: imageLink });
                                                    }
                                                });
                                        },
                                    );
                                } else {
                                    recordCategory("background");

                                    chrome.tabs.query({}, (tabs) => {
                                        tabs.forEach((tab) => {
                                            chrome.tabs
                                                .sendMessage(tab.id, {
                                                    action: "revealImage",
                                                    imageLink,
                                                })
                                                .catch((error) => {
                                                    // console.error(
                                                    //     `Error revealing image from URL (${imageLink}):`,
                                                    //     error,
                                                    // );
                                                });
                                        });
                                    });
                                }

                                categoryCount[className] =
                                    (categoryCount[className] || 0) + 1;
                            } else {
                                chrome.tabs.query({}, (tabs) => {
                                    tabs.forEach((tab) => {
                                        chrome.tabs
                                            .sendMessage(tab.id, {
                                                action: "revealImage",
                                                imageLink,
                                            })
                                            .catch((error) => {
                                                // console.error(
                                                //     `Error revealing image from URL (${imageLink}):`,
                                                //     error,
                                                // );
                                            });
                                    });
                                });

                                categoryCount["background"] =
                                    (categoryCount["background"] || 0) + 1;
                            }
                        } else {
                            console.log(
                                `URL: ${imageLink} | Prediction: background`,
                            );
                            categoryCount["background"] =
                                (categoryCount["background"] || 0) + 1;
                        }
                    });
                } catch (error) {
                    console.error(
                        `Error getting predictions from URL (${imageLink}):`,
                        error,
                    );
                }
            },
        );

        await Promise.all(predictionPromises);

        console.log("Category counts:");
        Object.entries(categoryCount).forEach(([category, count]) => {
            console.log(`${category}: ${count}`);
        });
    } else if (request.texts) {
        console.log(request.texts.length, "text to process");
        const categoryCount = {};

        // Keep existing sentence splitting for accuracy
        const allSentences = request.texts.flatMap(rawText => {
            return rawText
                .split(/(?<=[.!?])\s+/)
                .map(s => s.trim())
                .filter(s => s.length > 0)
        });

        console.log("sentences", allSentences);

        if (allSentences.length === 0) return;

        // Read confidence threshold once
        const { confidence: storedConfidence } = await chrome.storage.local.get(["confidence"]);
        const confidenceThreshold = storedConfidence ?? 0.5;

        // Preload content-category toggles once
        const categories = {
            profanity: "profanity",
            explicit: "explicit-content",
            drugs: "drugs",
            games: "web-based-games",
            gambling: "gambling",
            background: "background",
        };
        const categoryKeys = Object.values(categories);
        const categoryToggles = await chrome.storage.local.get(categoryKeys);

        // Batch sentences to reduce API calls
        const BATCH_SIZE = 50;
        const batches = chunk(allSentences, BATCH_SIZE);

        for (const batch of batches) {
            try {
                const suppFormData = { texts: batch };
                const suppResponse = await fetch(suppTextUrl, {
                    method: "POST",
                    body: JSON.stringify(suppFormData),
                    headers: { "Content-Type": "application/json" },
                });

                const suppPredictions = await suppResponse.json(); // [{ text, flags: [{category, confidence}]}]

                // Process each text result
                for (const result of suppPredictions) {
                    const text = result.text;
                    const flags = Array.isArray(result.flags) ? result.flags : [];

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
                        if (conf > confidenceThreshold && localCategoryCounter.hasOwnProperty(cat)) {
                            localCategoryCounter[cat] += 1;
                        }
                    });

                    // Pick category with most flags above threshold
                    let maxFlagged = null;
                    let maxCount = 0;
                    for (const [cat, cnt] of Object.entries(localCategoryCounter)) {
                        if (cnt > maxCount) {
                            maxFlagged = cat;
                            maxCount = cnt;
                        }
                    }

                    if (!maxFlagged) {
                        console.log(`Text: ${text} | Prediction: background`);
                        categoryCount["background"] = (categoryCount["background"] || 0) + 1;
                        continue;
                    }

                    const className = maxFlagged;
                    const storageKey = categories[className];
                    const isEnabled = categoryToggles[storageKey] ?? true; // default-on behavior

                    if (isEnabled) {
                        console.log(`Text ${text} | Prediction: ${className}`);
                        chrome.tabs.query({}, (tabs) => {
                            tabs.forEach((tab) => {
                                chrome.tabs.sendMessage(tab.id, {
                                    action: "removeText",
                                    text,
                                }).catch(() => {});
                            });
                        });
                        categoryCount[className] = (categoryCount[className] || 0) + 1;
                    } else {
                        // Disabled category: treat as background for UI purposes
                        categoryCount["background"] = (categoryCount["background"] || 0) + 1;
                    }
                }
            } catch (error) {
                console.error(`Error getting predictions`, error);
            }
        }

        console.log("Category counts:");
        Object.entries(categoryCount).forEach(([category, count]) => {
            console.log(`${category}: ${count}`);
        });
    }
});

// Small chunk helper for batching
function chunk(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
}
