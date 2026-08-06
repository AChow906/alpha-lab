UV := env -u UV_CONSTRAINT -u UV_BUILD_CONSTRAINT UV_DEFAULT_INDEX=https://pypi.org/simple uv

.PHONY: lock sync add add-dev check-lock lint fmt fmt-check test check

lock:
	$(UV) lock

sync:
	$(UV) sync

add:
	$(UV) add $(pkg)

add-dev:
	$(UV) add --dev $(pkg)

check-lock:
	@if grep -q "build.hubteam.com" uv.lock; then \
		echo "ERROR: uv.lock has internal HubSpot URLs — re-run 'make lock'"; exit 1; \
	else echo "uv.lock clean (public PyPI only)"; fi

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .

fmt-check:
	$(UV) run ruff format --check .

test:
	$(UV) run pytest

check: check-lock lint fmt-check test
