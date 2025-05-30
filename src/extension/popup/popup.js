document.getElementById('toggle').addEventListener('click', () => {
  chrome.runtime.sendMessage({ command: 'toggle' }, response => {
    if (chrome.runtime.lastError) {
      console.error('Toggle error:', chrome.runtime.lastError);
      document.getElementById('status').textContent = 'Status: Error';
    } else {
      document.getElementById('status').textContent = `Status: ${response.status}`;
    }
  });
});

document.getElementById('search').addEventListener('click', () => {
  chrome.runtime.sendMessage({ command: 'search', query: 'test' }, response => {
    if (chrome.runtime.lastError) {
      console.error('Search error:', chrome.runtime.lastError);
      document.getElementById('status').textContent = 'Status: Error';
    } else {
      document.getElementById('status').textContent = `Data: ${response.data.substring(0, 50)}...`;
    }
  });
});
