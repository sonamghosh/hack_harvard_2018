chrome.browserAction.onClicked.addListener(function(tab) {
    getSelectedText(tab.id, function(text) {
        localStorage.selectedText = text;
    });
});

function getSelectedText(tabId, cb) {
    chrome.tabs.executeScript(tabId, {
        code: "window.getSelection().toString();"
    });
}
