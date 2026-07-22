import json

from usage_tracker import extract_usage_from_json, extract_usage_from_sse_line
from proxy import _resolve_path, calculate_cost
from models_config import validate_model_form, save_model


class TestPathResolution:
    def test_appends_path_to_base(self):
        assert (
            _resolve_path("/v1/chat/completions", "https://api.example.com/inference/v1")
            == "https://api.example.com/inference/v1/chat/completions"
        )

    def test_drops_duplicate_segment(self):
        assert (
            _resolve_path("/v1/chat/completions", "https://api.example.com/inference/v1")
            == "https://api.example.com/inference/v1/chat/completions"
        )

    def test_plain_base_url(self):
        assert (
            _resolve_path("/v1/chat/completions", "https://api.example.com")
            == "https://api.example.com/v1/chat/completions"
        )

    def test_no_incoming_path(self):
        assert _resolve_path("", "https://api.example.com/v1") == "https://api.example.com/v1"


class TestUsageExtraction:
    def test_openai_format(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        usage = extract_usage_from_json(data)
        assert usage == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_tokens": 0,
        }

    def test_cached_tokens_details(self):
        data = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 5},
            }
        }
        usage = extract_usage_from_json(data)
        assert usage["cached_tokens"] == 5

    def test_gemini_format(self):
        data = {
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "cachedContentTokenCount": 2,
            }
        }
        usage = extract_usage_from_json(data)
        assert usage == {
            "input_tokens": 5,
            "output_tokens": 7,
            "cached_tokens": 2,
        }

    def test_sse_line(self):
        line = json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
        usage = extract_usage_from_sse_line(line)
        assert usage == {
            "input_tokens": 1,
            "output_tokens": 2,
            "cached_tokens": 0,
        }

    def test_done_line_returns_none(self):
        assert extract_usage_from_sse_line("[DONE]") is None


class TestCostCalculation:
    def test_remote_cost(self):
        model = {
            "type": "remote",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            "cached_price_per_million": 0.0,
        }
        usage = {
            "input_tokens": 1_000_000,
            "output_tokens": 500_000,
            "cached_tokens": 0,
        }
        cost = calculate_cost(model, usage, 1.0, {})
        assert cost == 2.0

    def test_remote_with_cached_tokens(self):
        model = {
            "type": "remote",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            "cached_price_per_million": 0.1,
        }
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 1_000_000,
        }
        cost = calculate_cost(model, usage, 0, {})
        assert cost == 0.1

    def test_remote_cached_tokens_not_double_counted(self):
        """
        Regression test: input_tokens from OpenAI includes cached tokens,
        so they must not be double-counted.
        """
        model = {
            "type": "remote",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            "cached_price_per_million": 0.1,
        }
        usage = {
            "input_tokens": 1_500_000,    # includes 1M cached
            "output_tokens": 500_000,
            "cached_tokens": 1_000_000,   # subset of input_tokens
        }
        cost = calculate_cost(model, usage, 0, {})
        # Expected: (500k non-cached input * $1.0/M) + (500k output * $2.0/M) + (1M cached * $0.1/M)
        # = $0.50 + $1.00 + $0.10 = $1.60
        assert cost == 1.60, f"Expected $1.60, got ${cost}"

    def test_local_cost(self):
        model = {"type": "local"}
        usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        settings = {
            "electricity_cost_per_kwh": 0.12,
            "local_model_max_wattage": 300,
        }
        # 1 hour duration
        cost = calculate_cost(model, usage, 3600.0, settings)
        assert round(cost, 4) == 0.0360


class TestModelValidation:
    def test_valid_remote(self):
        form = {
            "name": "test-model",
            "base_url": "https://api.example.com",
            "api_model_name": "gpt-4",
            "type": "remote",
            "input_price_per_million": "1.00",
            "output_price_per_million": "2.00",
            "cached_price_per_million": "0.50",
        }
        errors = validate_model_form(form, editing=False)
        assert errors == []

    def test_missing_name(self):
        form = {
            "name": "   ",
            "base_url": "https://api.example.com",
            "api_model_name": "x",
            "type": "remote",
            "input_price_per_million": "1",
            "output_price_per_million": "1",
        }
        errors = validate_model_form(form, editing=False)
        assert any("Name is required" in e for e in errors)

    def test_invalid_name_chars(self):
        form = {
            "name": "bad name!",
            "base_url": "https://api.example.com",
            "api_model_name": "x",
            "type": "remote",
            "input_price_per_million": "1",
            "output_price_per_million": "1",
        }
        errors = validate_model_form(form, editing=False)
        assert any("letters, numbers, hyphens, and underscores" in e for e in errors)

    def test_negative_price(self):
        form = {
            "name": "test",
            "base_url": "https://api.example.com",
            "api_model_name": "x",
            "type": "remote",
            "input_price_per_million": "-1",
            "output_price_per_million": "1",
        }
        errors = validate_model_form(form, editing=False)
        assert any("must be >= 0" in e for e in errors)


class TestFlaskApp:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "models" in data

    def test_models_list_includes_aliases(self, client):
        save_model({
            "name": "fast-model",
            "display_name": "Fast Model",
            "provider": "test",
            "type": "remote",
            "tags": ["fast"],
            "base_url": "https://api.example.com",
            "api_key": "test-key",
            "api_model_name": "test-model",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
            "cached_price_per_million": 0.0,
            "enabled": True,
        })

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.get_json()
        ids = {m["id"] for m in data["data"]}
        assert "fast-model" in ids
        assert "fast" in ids

    def test_settings_page(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Proxy Timeout" in resp.get_data(as_text=True)

    def test_settings_update(self, client):
        resp = client.post("/settings", data={
            "electricity_cost_per_kwh": "0.15",
            "local_model_max_wattage": "400",
            "proxy_timeout_seconds": "60",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert "Settings saved" in resp.get_data(as_text=True)

    def test_proxy_open_when_no_key(self, client):
        # Default: PROXY_API_KEY unset → /v1/* is not gated.
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_proxy_api_key_enforced(self, client, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "PROXY_API_KEY", "s3cret")

        # Missing key → 401
        assert client.get("/v1/models").status_code == 401
        # Wrong key → 401
        assert client.get(
            "/v1/models", headers={"Authorization": "Bearer nope"}
        ).status_code == 401
        # Correct key → 200
        assert client.get(
            "/v1/models", headers={"Authorization": "Bearer s3cret"}
        ).status_code == 200

    def test_web_ui_not_gated_by_proxy_key(self, client, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "PROXY_API_KEY", "s3cret")
        # The web UI / health check are never gated by the proxy key.
        assert client.get("/health").status_code == 200
