.PHONY: all clean docker

all:
	@SASS_ARCH?=sm_80
	@echo "Building for $${SASS_ARCH}"
	@SASS_ARCH=$${SASS_ARCH} bash scripts/build.sh

docker:
	@bash scripts/docker-build.sh

clean:
	@bash scripts/clean.sh
