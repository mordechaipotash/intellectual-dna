/* brain-mcp dashboard — Deep Space Neural JS */

// ═══════════════════════════════════════════════════════════════
// COUNTER ANIMATION
// ═══════════════════════════════════════════════════════════════

function animateCounter(el, target, duration) {
    if (duration === undefined) duration = 2000;
    var start = 0;
    var step = function(ts) {
        if (!start) start = ts;
        var progress = Math.min((ts - start) / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(eased * target).toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

/**
 * Find all elements with data-counter attribute and animate them.
 * Called after htmx swaps in stats content.
 */
function initCounters(root) {
    if (!root) root = document;
    var counters = root.querySelectorAll('[data-counter]');
    counters.forEach(function(el) {
        var target = parseInt(el.getAttribute('data-counter'), 10);
        if (!isNaN(target) && target > 0) {
            el.textContent = '0';
            animateCounter(el, target);
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// HTMX EVENT HOOKS — trigger counter animation after swap
// ═══════════════════════════════════════════════════════════════

document.addEventListener('htmx:afterSwap', function(evt) {
    initCounters(evt.detail.target);
});

// Also run on initial page load (for any pre-rendered counters)
document.addEventListener('DOMContentLoaded', function() {
    initCounters();
});

// ═══════════════════════════════════════════════════════════════
// SEARCH DEBOUNCE (500ms)
// ═══════════════════════════════════════════════════════════════

var searchTimer = null;
function debounceSearch(el, url) {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function() {
        htmx.ajax('GET', url + '?q=' + encodeURIComponent(el.value), '#search-results');
    }, 500);
}

// ═══════════════════════════════════════════════════════════════
// SSE HELPER — for long-running tasks
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
// HEATMAP RENDERER (365-day calendar grid on <canvas>)
// ═══════════════════════════════════════════════════════════════

function renderHeatmap(canvasId, data, days) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var cellSize = 12;
    var gap = 2;
    var step = cellSize + gap;

    // Build a date->count lookup
    var lookup = {};
    var maxCount = 1;
    data.forEach(function(d) {
        lookup[d.date] = d.count;
        if (d.count > maxCount) maxCount = d.count;
    });

    // Calculate grid: 7 rows (days of week), N columns (weeks)
    var today = new Date();
    today.setHours(0, 0, 0, 0);
    var startDate = new Date(today);
    startDate.setDate(startDate.getDate() - days + 1);
    // Align to start of week (Sunday)
    var dayOfWeek = startDate.getDay();

    var cols = Math.ceil((days + dayOfWeek) / 7);
    canvas.width = cols * step + 2;
    canvas.height = 7 * step + 2;
    canvas.style.maxWidth = canvas.width + 'px';

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var currentDate = new Date(startDate);
    currentDate.setDate(currentDate.getDate() - dayOfWeek); // align to Sunday

    for (var col = 0; col < cols; col++) {
        for (var row = 0; row < 7; row++) {
            var dateStr = currentDate.toISOString().slice(0, 10);
            var count = lookup[dateStr] || 0;
            var x = col * step + 1;
            var y = row * step + 1;

            if (currentDate > today || currentDate < startDate) {
                ctx.fillStyle = 'rgba(30, 30, 50, 0.3)';
            } else if (count === 0) {
                ctx.fillStyle = 'rgba(30, 30, 50, 0.6)';
            } else {
                var intensity = Math.min(count / maxCount, 1);
                // Gradient: low=dim accent, high=bright accent
                var r = Math.round(99 + (129 - 99) * intensity);
                var g = Math.round(102 + (140 - 102) * intensity);
                var b = Math.round(241 + (248 - 241) * intensity);
                var a = 0.3 + intensity * 0.7;
                ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
            }

            ctx.fillRect(x, y, cellSize, cellSize);
            currentDate.setDate(currentDate.getDate() + 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// SSE HELPER — for long-running tasks
// ═══════════════════════════════════════════════════════════════

function connectSSE(taskId, targetId) {
    var source = new EventSource('/api/tasks/' + taskId + '/stream');
    var target = document.getElementById(targetId);

    source.onmessage = function(event) {
        var data = JSON.parse(event.data);
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
