/* brain-mcp dashboard — custom JavaScript */

// Search debounce (500ms)
let searchTimer = null;
function debounceSearch(el, url) {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
        htmx.ajax('GET', url + '?q=' + encodeURIComponent(el.value), '#search-results');
    }, 500);
}

// SSE helper for long-running tasks
function connectSSE(taskId, targetId) {
    const source = new EventSource('/api/tasks/' + taskId + '/stream');
    const target = document.getElementById(targetId);

    source.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (target) {
            target.innerHTML = data.message || data.progress || '';
        }
        if (data.status === 'done' || data.status === 'failed') {
            source.close();
            // Refresh stats after task completes
            htmx.ajax('GET', '/api/stats/overview', '#stats-cards');
        }
    };

    source.onerror = function() {
        source.close();
    };
}
