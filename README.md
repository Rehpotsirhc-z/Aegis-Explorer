# Chrome-Extension

## Roadmap

-   [ ] Fix the weird https workaround in line 212 of contentScript
-   [ ] Move the background to inside the webrequest
-   [ ] Train the text AI without monetary and social and remove all of those from every file
-   [ ] Fix the contentscript webrequest image to hide the images
-   [X] Make text log to pie chart
-   [X] Figure out whats up with the chart of active extension
-   [X] Make sure that active extension works as intended
-   [-] Cache for the text and images
-   [ ] Figure out line 221 of `settings.js` why there is that error


Image Links Workflow
1. ContentScript hides every image link it can
    -Background gives the list of images found during request and we can hide if they are found in the HTML
    -*separately, the ContentScript uses it's existing logic too
2. Background, once it sends and get's approval by the server, sends a message to content script
3. ContentScript, being told which URL is ok, track the original position in the HTML and re-enables