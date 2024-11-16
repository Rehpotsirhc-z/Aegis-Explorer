const style = document.createElement("style");
style.textContent = `
    img:not(.approved) {
        display: none !important;
    }
    *:not(.approved) {
        background-image: none !important;
    }
`;
document.documentElement.appendChild(style);

var urlStatuses;

const seenImages = new Set();
const seenText = new Set();

// set keeping track of image URLs
const imageUrls = new Set();

// chrome.webRequest.onBeforeRequest.addListener(
//     function (details) {
//         if (details.type === "image") {
//             console.log("Image request:", details.url);

//             // check if image url is in the set. If not, fetch request and add to the set
//             if (!imageUrls.has(details.url)) {
//                 console.log("New image URL:", details.url);
//                 // send message with request.images to trigger processing
//                 // chrome.runtime.sendMessage({ images: [details.url] });
//             }

//             imageUrls.add(details.url);

//         }
//     },
//     { urls: ["<all_urls>"] },
// );

function revealImage(imageLink) {
        var revealed = false;

        const images = document.querySelectorAll("img");

        filteredImageLink = imageLink.replace(/^https?:\/\//, "");

        images.forEach((element) => {
            // go through attributes
            // console.log(element.attributes);
            for (let attr of element.attributes) {
                if (attr.value.includes(filteredImageLink)) { // TODO Fix this
                    revealed = true;
                    element.classList.add("approved");
                }
            }
        })

        if (!revealed) {
            console.log("Failed to reveal image", imageLink);
        } else {
            console.log("Revaled image", imageLink);
        }
}

function extractImageLinks() {
    const images = document.querySelectorAll("img");

    const newImageLinks = Array.from(images)
        .filter((img) => img.dataset.approved !== "true")
        .map((img) => {
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
            // // TODO
            img.alt = "";

            return img.dataset.originalSrc;
        })
        .filter((src) => !seenImages.has(src)); // We do this after so that they still disappear if not approved

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
                console.error("Error extracting background image");
            }
        }
    });

    console.log(`${newImageLinks.length} new images`);
    return newImageLinks;
}

// function sendImages() {
//     const imageLinks = extractImageLinks();
//     try {
//         if (imageLinks.length > 0) {
//             chrome.runtime.sendMessage({ images: imageLinks });
//         }
//     } catch (error) {
//         console.error("Error sending images", error);
//     }
// }

function extractSentences() {
    // const textContent = document.body.innerText;
    // const sentences = textContent.match(/[^.!?]*[.!?]/g) || [];
    // return sentences;
    sentences = [];
    function extractTextFromNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            textContent = node.textContent;
            if (textContent.trim() !== "") {
                sentences.push(node.textContent);
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
        .filter((sentence) => !seenText.has(sentence));

    return sentences;
}

function sendText() {
    const textLinks = extractSentences();
    seenText.add(textLinks.map((text) => text.trim()));
    console.log(seenText);

    console.log(textLinks);
    try {
        if (textLinks.length > 0) {
            chrome.runtime.sendMessage({ text: textLinks });
        }
    } catch (error) {
        console.error("Error sending text", error);
    }
}

// Set up a MutationObserver to detect change in the DOM
// const observer = new MutationObserver(() => {
//     sendImages();
//     sendText();
// });

// observer.observe(document.body, {
//     childList: true,
//     subtree: true,
//     //     // attributes: true,
//     //     // attributesFilter: ["src"],
// });

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    // if (message.images) {
    //     console.log("Received images", message.images);

    //     message.images.forEach((imageLink) => {
    //         console.log("Removing image", imageLink);

    //         // get everywhere that this image link occurs not just src
    //         const allElements = document.querySelectorAll('*');
    //         allElements.forEach((element) => {
    //             // go through attributes
    //             for (let attr of element.attributes) {
    //                 if (attr.value.includes(imageLink)) {
    //                     console.log("Removing image from attribute", attr, attr.value, imageLink, element);
    //                     element.setAttribute(attr.name, "");
    //                     console.log(attr);
    //                 }
    //             }
    //         })
    //     });
    // }
    if (message.urlStatuses) {
        // Assuming urlStatuses is a plain object now, you can use it directly
        urlStatuses = new Map(Object.entries(message.urlStatuses));
        console.log("URL statuses received", urlStatuses);

        // Convert Map to an array of entries, filter by status, and then process
        Array.from(urlStatuses.entries())
            .filter(([url, status]) => status === true)  // Filter by status
            .forEach(([url]) => {
                revealImage(url);  // Call revealImage for each approved URL
            });
    }
    if (message.action === "removeImage" && message.imageLink) {
        console.log("REMEMEMEOVMOVMO");
        // const imageLink = message.imageLink;

        // console.log("REMOVE", imageLink);

        // const images = document.querySelectorAll("img");

        // images.forEach((element) => {
        //     // go through attributes
        //     // console.log(element.attributes);
        //     for (let attr of element.attributes) {
        //         // console.log(attr);
        //         if (attr.value.includes(imageLink) || attr.value.includes(imageLink.substring(0, 5))) {
        //             element.classList.add("approved");
        //         }
        //     }
        // })

    } else if (message.action === "revealImage" && message.imageLink) {
        revealImage(message.imageLink);
    }

    // if (message.action === "removeImage" && message.imageLink) {
    //     console.log("Removing image", message.imageLink);

    //     const images = document.querySelectorAll(
    //         `img[data-original-src="${message.imageLink}"]`,
    //     );
    //     images.forEach((image) => {
    //         image.src = "";
    //         image.alt = "";
    //         if (image.srcset === "" && image.dataset.originalSrcset) {
    //             image.srcset = "";
    //             image.removeAttribute("data-original-srcset");
    //         }
    //         image.removeAttribute("data-original-src");
    //         image.removeAttribute("data-original-alt");
    //     });

    //     // Handle background images
    //     const elements = document.querySelectorAll(
    //         `*[data-original-background-image="${message.imageLink}"]`,
    //     );
    //     elements.forEach((element) => {
    //         element.style.backgroundImage = "none";
    //         element.removeAttribute("data-original-background-image");
    //     });
    // } else if (message.action === "revealImage" && message.imageLink) {
    //     console.log("Revealing image", message.imageLink);

    //     // reveal <img> elements
    //     const images  = document.querySelectorAll(`img[src="${message.imageLink}"]`);
    //     images.forEach((image) => {
    //         image.classList.add("approved");
    //     });

    //     // reveal background images
    //     const elements = document.querySelectorAll(`*[style*="${message.imageLink}"]`);
    //     elements.forEach((element) => {
    //         element.classList.add("approved");
    //     });

    //     // const images = document.querySelectorAll(
    //     //     `img[src=""][data-original-src="${message.imageLink}"]`,
    //     // );
    //     // images.forEach((image) => {
    //     //     image.src = image.dataset.originalSrc;
    //     //     image.alt = image.dataset.originalAlt;
    //     //     if (image.srcset === "" && image.dataset.originalSrcset) {
    //     //         image.srcset = image.dataset.originalSrcset;
    //     //         image.removeAttribute("data-original-srcset");
    //     //     }
    //     //     image.dataset.approved = "true";
    //     //     image.removeAttribute("data-original-src");
    //     //     image.removeAttribute("data-original-alt");
    //     // });

    //     // // Handle background images
    //     // const elements = document.querySelectorAll(
    //     //     `*[data-original-background-image="${message.imageLink}"]`,
    //     // );

    //     // elements.forEach((element) => {
    //     //     console.log("Revealing background image", message.imageLink);
    //     //     element.style.backgroundImage = `url(${element.dataset.originalBackgroundImage})`;
    //     //     element.dataset.approved = "true";
    //     //     element.removeAttribute("data-original-background-image");
    //     // });
    // } else if (message.action === "removeText" && message.text) {
    //     // remove all instances of the text in the document
    //     const text = message.text.trim();

    //     console.log("Removing text", text);

    //     function removeTextFromNode(node) {
    //         if (node.nodeType === Node.TEXT_NODE) {
    //             textContent = node.textContent;
    //             if (textContent.trim() === "") {
    //                 return;
    //             }
    //             // console.log("Target:", text);
    //             // console.log(node.textContent);
    //             // node.textContent = node.textContent.replace(text, "???");
    //             if (node.textContent.includes(text)) {
    //                 // console.log("Result: ", node.textContent);
    //                 console.log(
    //                     "new",
    //                     node.textContent.replace(new RegExp(text, "gi"), "???"),
    //                 );
    //                 node.textContent = node.textContent.replace(
    //                     new RegExp(text, "gi"),
    //                     "???",
    //                 );
    //                 console.log("Removoing: ", node.textContent);
    //             }
    //         } else {
    //             node.childNodes.forEach((child) => removeTextFromNode(child));
    //         }
    //     }

    //     removeTextFromNode(document.body);
    // }
    return true;
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

chrome.runtime.sendMessage({ action: "getUrlStatuses" });