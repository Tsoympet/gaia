function scrapePage() {
    const data = {
      text: document.body.innerText,
      title: document.title,
      url: window.location.href,
      metadata: {
        type: 'text',
        source: 'web',
        timestamp: new Date().toISOString()
      }
    };
    chrome.runtime.sendMessage({ command: 'data', payload: data });
}

function handleDynamicContent() {
    const observer = new MutationObserver(() => {
      scrapePage();
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

scrapePage();
handleDynamicContent();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.command === 'scrape') {
      scrapePage();
      sendResponse({ status: 'Scraped' });
    }
});
