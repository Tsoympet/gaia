document.getElementById('toggle').addEventListener('click', () => {
  chrome.runtime.sendMessage({ command: 'toggle' }, response => {
    document.getElementById('status').textContent = `Status: ${response.status}`;
  });
});

document.getElementById('search').addEventListener('click', () => {
  chrome.runtime.sendMessage({ command: 'search', query: 'test' }, response => {
    document.getElementById('status').textContent = `Data: ${response.data.substring(0, 50)}...`;
  });
});
