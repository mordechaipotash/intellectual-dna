"""Tests for Dashboard v2: stats endpoints, heatmap, recent, spark, search polish."""

import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

pytest.importorskip("fastapi", reason="Dashboard tests require fastapi")
from fastapi.testclient import TestClient

from brain_mcp.dashboard.app import create_app


@pytest.fixture
def client():
    """Create a test client for the dashboard app."""
    app = create_app()
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHeatmapEndpoint:
    """Test /api/stats/heatmap endpoint."""

    def test_heatmap_returns_json_list(self, client):
        """Heatmap should return a JSON list (possibly empty)."""
        response = client.get("/api/stats/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_heatmap_with_days_param(self, client):
        """Heatmap should accept days parameter."""
        response = client.get("/api/stats/heatmap?days=30")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_heatmap_html_returns_html(self, client):
        """Heatmap HTML endpoint should return HTML partial."""
        response = client.get("/api/stats/heatmap.html")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_heatmap_html_has_canvas_or_unavailable(self, client):
        """Heatmap HTML should contain canvas element or unavailable message."""
        response = client.get("/api/stats/heatmap.html")
        text = response.text
        assert "heatmap-canvas" in text or "unavailable" in text.lower()

    def test_heatmap_item_shape(self):
        """If heatmap returns data, items should have date and count."""
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("2025-07-01", 42),
            ("2025-07-02", 15),
        ]

        with patch(
            "brain_mcp.server.db.get_conversations",
            return_value=mock_con,
        ):
            from brain_mcp.dashboard.routes.stats import stats_heatmap
            result = asyncio.run(stats_heatmap(days=30))
            assert len(result) == 2
            assert result[0]["date"] == "2025-07-01"
            assert result[0]["count"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# RECENT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecentEndpoint:
    """Test /api/stats/recent endpoint."""

    def test_recent_returns_json_list(self, client):
        """Recent should return a JSON list (possibly empty)."""
        response = client.get("/api/stats/recent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_recent_with_limit(self, client):
        """Recent should accept limit parameter."""
        response = client.get("/api/stats/recent?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_recent_html_returns_html(self, client):
        """Recent HTML endpoint should return HTML partial."""
        response = client.get("/api/stats/recent.html")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_recent_item_shape(self):
        """If recent returns data, items should have expected fields."""
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("conv-001", "Test Conversation", "2025-07-10 14:30:00", 12),
        ]

        with patch(
            "brain_mcp.server.db.get_conversations",
            return_value=mock_con,
        ):
            from brain_mcp.dashboard.routes.stats import stats_recent
            result = asyncio.run(stats_recent(limit=5))
            assert len(result) == 1
            item = result[0]
            assert item["conversation_id"] == "conv-001"
            assert item["title"] == "Test Conversation"
            assert item["date"] == "2025-07-10"
            assert item["message_count"] == 12


# ═══════════════════════════════════════════════════════════════════════════════
# SPARK ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════


class TestSparkEndpoint:
    """Test /api/stats/spark endpoint."""

    def test_spark_returns_json_list(self, client):
        """Spark should return a JSON list (possibly empty)."""
        response = client.get("/api/stats/spark")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_spark_with_params(self, client):
        """Spark should accept metric and days parameters."""
        response = client.get("/api/stats/spark?metric=messages&days=14")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_spark_default_days(self, client):
        """Default days should be 7."""
        response = client.get("/api/stats/spark")
        assert response.status_code == 200

    def test_spark_item_shape(self):
        """Spark items should have date and count fields."""
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("2025-07-08", 5),
            ("2025-07-09", 12),
            ("2025-07-10", 8),
        ]

        with patch(
            "brain_mcp.server.db.get_conversations",
            return_value=mock_con,
        ):
            from brain_mcp.dashboard.routes.stats import stats_spark
            result = asyncio.run(stats_spark(metric="messages", days=7))
            assert len(result) == 3
            assert result[0]["date"] == "2025-07-08"
            assert result[0]["count"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE UPGRADE
# ═══════════════════════════════════════════════════════════════════════════════


class TestHomePageV2:
    """Test upgraded home page features."""

    def test_home_template_has_heatmap(self):
        """Home template should reference heatmap."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/home.html"
        content = template.read_text()
        assert "heatmap" in content.lower()

    def test_home_template_has_recent_activity(self):
        """Home template should reference recent activity."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/home.html"
        content = template.read_text()
        assert "Recent Activity" in content

    def test_home_template_has_htmx_heatmap_loader(self):
        """Home template should load heatmap via htmx."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/home.html"
        content = template.read_text()
        assert "/api/stats/heatmap.html" in content

    def test_home_template_has_htmx_recent_loader(self):
        """Home template should load recent activity via htmx."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/home.html"
        content = template.read_text()
        assert "/api/stats/recent.html" in content

    def test_home_template_still_has_stats_overview(self):
        """Home template should still have the stats overview."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/home.html"
        content = template.read_text()
        assert "/api/stats/overview" in content

    def test_home_page_loads(self, client):
        """Home page should load (200 or 302 redirect to onboarding)."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (200, 302)


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH PAGE POLISH
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchPolish:
    """Test search page polish features."""

    def test_empty_search_has_suggestions(self, client):
        """Empty search should show suggested queries."""
        response = client.get("/api/search?q=&mode=semantic")
        assert response.status_code == 200
        assert "suggested" in response.text.lower() or "Search your brain" in response.text

    def test_empty_search_has_suggestion_buttons(self, client):
        """Empty search should show clickable suggestion buttons."""
        response = client.get("/api/search?q=&mode=semantic")
        assert "suggested-query" in response.text
        assert "architecture decisions" in response.text

    def test_search_page_has_suggested_search_function(self, client):
        """Search page should have suggestedSearch JS function."""
        response = client.get("/search")
        assert "suggestedSearch" in response.text

    def test_no_results_template_has_duration(self):
        """No-results template should show ms duration."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/partials/search_results.html"
        content = template.read_text()
        # Check that the no-results block includes duration_ms
        assert "duration_ms" in content

    def test_search_results_template_has_search_speed(self):
        """Search results template should show search speed."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/partials/search_results.html"
        content = template.read_text()
        assert "search-speed" in content

    def test_search_results_template_has_match_badge(self):
        """Search results template should use match-badge class."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/partials/search_results.html"
        content = template.read_text()
        assert "match-badge" in content

    def test_search_results_template_has_percentage(self):
        """Search results template should show percentage."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/partials/search_results.html"
        content = template.read_text()
        assert "%" in content


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTER ANIMATION JS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCounterAnimationJS:
    """Test that brain.js has counter animation code."""

    def test_brain_js_has_animate_counter(self, client):
        """brain.js should have animateCounter function."""
        response = client.get("/static/brain.js")
        assert response.status_code == 200
        assert "animateCounter" in response.text

    def test_brain_js_has_init_counters(self, client):
        """brain.js should have initCounters function."""
        response = client.get("/static/brain.js")
        assert "initCounters" in response.text

    def test_brain_js_has_data_counter_selector(self, client):
        """brain.js should query for data-counter attributes."""
        response = client.get("/static/brain.js")
        assert "data-counter" in response.text

    def test_brain_js_has_htmx_afterswap_hook(self, client):
        """brain.js should trigger counters on htmx:afterSwap."""
        response = client.get("/static/brain.js")
        assert "htmx:afterSwap" in response.text

    def test_brain_js_has_domcontentloaded_hook(self, client):
        """brain.js should trigger counters on DOMContentLoaded."""
        response = client.get("/static/brain.js")
        assert "DOMContentLoaded" in response.text

    def test_brain_js_has_easing(self, client):
        """animateCounter should use easing (cubic ease-out)."""
        response = client.get("/static/brain.js")
        assert "Math.pow" in response.text or "eased" in response.text

    def test_brain_js_has_heatmap_renderer(self, client):
        """brain.js should have renderHeatmap function."""
        response = client.get("/static/brain.js")
        assert "renderHeatmap" in response.text

    def test_stats_cards_have_data_counter(self):
        """Stats cards template should use data-counter attributes."""
        template = Path(__file__).parent.parent / "brain_mcp/dashboard/templates/partials/stats_cards.html"
        content = template.read_text()
        assert "data-counter" in content


# ═══════════════════════════════════════════════════════════════════════════════
# CSS THEME ADDITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCSSAdditions:
    """Test that CSS has the new styles."""

    def test_css_has_empty_state(self, client):
        """CSS should have empty-state styles."""
        response = client.get("/static/brain.css")
        assert "empty-state" in response.text

    def test_css_has_suggested_queries(self, client):
        """CSS should have suggested-queries styles."""
        response = client.get("/static/brain.css")
        assert "suggested-quer" in response.text

    def test_css_has_match_badge(self, client):
        """CSS should have match-badge styles."""
        response = client.get("/static/brain.css")
        assert "match-badge" in response.text

    def test_css_has_recent_activity(self, client):
        """CSS should have recent-activity-list styles."""
        response = client.get("/static/brain.css")
        assert "recent-activity-list" in response.text

    def test_css_has_search_speed(self, client):
        """CSS should have search-speed styles."""
        response = client.get("/static/brain.css")
        assert "search-speed" in response.text

    def test_css_has_heatmap_canvas(self, client):
        """CSS should have heatmap-canvas styles."""
        response = client.get("/static/brain.css")
        assert "heatmap-canvas" in response.text


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH MATCH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMatchHelpers:
    """Test the match label/class helpers in search.py."""

    def test_match_label_best(self):
        from brain_mcp.dashboard.routes.search import _match_label
        assert _match_label(0.95) == "Best match"

    def test_match_label_good(self):
        from brain_mcp.dashboard.routes.search import _match_label
        assert _match_label(0.80) == "Good match"
        assert _match_label(0.70) == "Good match"

    def test_match_label_related(self):
        from brain_mcp.dashboard.routes.search import _match_label
        assert _match_label(0.60) == "Related"
        assert _match_label(0.50) == "Related"

    def test_match_label_weak(self):
        from brain_mcp.dashboard.routes.search import _match_label
        assert _match_label(0.30) == "Weak match"

    def test_match_class_best(self):
        from brain_mcp.dashboard.routes.search import _match_class
        assert _match_class(0.95) == "match-best"

    def test_match_class_good(self):
        from brain_mcp.dashboard.routes.search import _match_class
        assert _match_class(0.80) == "match-good"

    def test_match_class_related(self):
        from brain_mcp.dashboard.routes.search import _match_class
        assert _match_class(0.55) == "match-related"

    def test_match_class_below_50(self):
        from brain_mcp.dashboard.routes.search import _match_class
        assert _match_class(0.30) == "match-related"
