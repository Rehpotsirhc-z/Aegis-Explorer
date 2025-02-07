# Chrome-Extension

## Roadmap

-   [X] Text AI use confidence
-   [X] STRT Text sending too much
-   [X] Set the age to 3--5
-   [ ] Fix the data images changing themselves to the URL of the website (extract running more than once)
-   [-] Fix the text AI being so slow
-   [ ] Retrain text AI

-   [X] Train the text AI without monetary and social and remove all of those from every file
-   [X] Make text log to pie chart
-   [X] Figure out whats up with the chart of active extension
-   [X] Make sure that active extension works as intended
-   [-] Cache for the text and images
-   [ ] Figure out line 221 of `settings.js` why there is that error

-   [ ] CANCELED? Fix the weird https workaround in line 212 of contentScript
-   [ ] CANCELED Move the background to inside the webrequest
-   [ ] CANCELED Fix the contentscript webrequest image to hide the images
-   [ ] CANCELED Fix the declarative net request failing with lazy placeholder
-   [ ] CANCELED Fix the failed to show with urlStatuses


Image Links Workflow
1. ContentScript hides every image link it can
    -Background gives the list of images found during request and we can hide if they are found in the HTML
    -*separately, the ContentScript uses it's existing logic too
2. Background, once it sends and get's approval by the server, sends a message to content script
3. ContentScript, being told which URL is ok, track the original position in the HTML and re-enables
