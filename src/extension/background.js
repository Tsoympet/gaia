let ws;
let enabled = true;

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8766');
    ws.onopen = () => console.log('Connected to G.A.I.A core');
    ws.onmessage = handleMessage;
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 5000);
    };
    ws.onerror = error => console.error('WebSocket error:', error);
}

function handleMessage(event) {
    try {
        const data = JSON.parse(event.data);
        if (data.command === 'search' && enabled) {
            fetchData(data.payload).then(result => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'universal_data', payload: result }));
                }
            });
        } else if (data.command === 'toggle') {
            enabled = !enabled;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'status', payload: enabled }));
            }
        } else if (data.command === 'share') {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ 
                    type: 'shared_knowledge_data', 
                    faction: data.faction, 
                    payload: data.payload 
                }));
            }
        }
    } catch (error) {
        console.error('Error handling WebSocket message:', error);
    }
}

async function fetchData(query) {
    try {
        // Placeholder: Replace with actual API or scraping logic
        const response = await fetch(`https://api.example.com/search?q=${encodeURIComponent(query)}`, {
            headers: { 'User-Agent': navigator.userAgent }
        });
        const text = await response.text();
        return { text, metadata: { source: 'web', type: 'text' } };
    } catch (error) {
        console.error('Fetch error:', error);
        return { text: '', metadata: { error: error.message } };
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    try {
        if (request.command === 'toggle') {
            enabled = !enabled;
            sendResponse({ status: enabled ? 'Enabled' : 'Disabled' });
        } else if (request.command === 'search') {
            fetchData(request.query).then(result => {
                sendResponse({ data: result.text });
            });
            return true; // Keep message channel open for async response
        } else if (request.command === 'data') {
            if (enabled && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'universal_data', payload: request.payload }));
            }
            sendResponse({ status: 'Data received' });
        }
    } catch (error) {
        console.error('Error handling runtime message:', error);
        sendResponse({ status: 'Error', error: error.message });
    }
    return true; // Ensure async responses are supported
});

connectWebSocket();
