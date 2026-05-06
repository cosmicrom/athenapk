COLOR_RESET=\033[0m
BOLD=\033[1m
BLUE_ARROW=\033[0;34m \033[0m

machine ?= native

BUILD_DIR := $(if $(filter native, $(machine)), build, build-$(machine))

.PHONY: build

build:
	@echo "${BLUE_ARROW}${BOLD}Configuring Build Directory - Machine Config: $(machine)${COLOR_RESET}"
	cmake -S . -B $(BUILD_DIR) -DMACHINE_CFG=cmake/$(machine).cmake
	@echo "\n${BLUE_ARROW}${BOLD}Building AthenaPK${COLOR_RESET}"
	cmake --build $(BUILD_DIR) --parallel 8
	@echo "${BLUE_ARROW}${BOLD}AthenaPK Built${COLOR_RESET}"

clean:
	rm -rf build*
