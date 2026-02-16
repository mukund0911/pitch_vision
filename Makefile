.PHONY: install test process-video serve

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short --cov=pitchvision

# Video → tracking JSON (requires --video, --homography or --calibrate, --output)
process-video:
	python scripts/process_video.py \
		--video $(VIDEO) \
		--homography $(HOMOGRAPHY) \
		--output $(OUT) \
		$(if $(SUBSAMPLE),--subsample $(SUBSAMPLE),) \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(MAX_FRAMES),--max-frames $(MAX_FRAMES),)

serve:
	uvicorn server.main:app --reload --port 8000
