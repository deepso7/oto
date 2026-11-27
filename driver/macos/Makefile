# Builds OtoMicrophone.driver (universal arm64 + x86_64), the oto-verify tool and the signed
# oto-hal-helper (driver install/uninstall, see tools/oto-hal-helper.c) into build/.
#
#   make                          build and sign ad-hoc (local development)
#   SIGN_ID="<identity>" make     build and sign with that keychain identity
#   SIGN_ID="Developer ID Application: Name (TEAMID)" make release
#                                 build for distribution: hardened runtime + secure timestamp;
#                                 fails unless SIGN_ID is a Developer ID identity in the keychain
#   make test                     exercise oto-hal-helper's lock/swap/naming in a scratch dir
#   make clean
#
# SIGN_ID is a certificate name or SHA-1 hash from `security find-identity -v -p codesigning`.
# No identity lives in the repo: it always comes from the environment or the command line.
# Run as yourself, never as root. scripts/macos/install-driver.sh installs the result.

BUILD    := build
BUNDLE   := $(BUILD)/OtoMicrophone.driver
BIN      := $(BUNDLE)/Contents/MacOS/OtoMicrophone
STAMP    := $(BUILD)/.signed
VERIFY   := $(BUILD)/oto-verify
HELPER   := $(BUILD)/oto-hal-helper
HSTAMP   := $(BUILD)/.helper-signed
ARCHS    := -arch arm64 -arch x86_64
MINOS    := -mmacosx-version-min=12.0
CFLAGS   := -std=c11 -O2 -Wall -Wextra -Werror $(ARCHS) $(MINOS)
SIGN_ID  ?= -
ifeq ($(strip $(SIGN_ID)),)
override SIGN_ID := -
endif

# Developer ID builds (and everything under `make release`) get the hardened runtime and a
# secure timestamp, which notarization requires. Other identities skip the timestamp server.
ifneq ($(or $(RELEASE),$(findstring Developer ID Application,$(SIGN_ID))),)
SIGN_FLAGS := --options runtime --timestamp
else
SIGN_FLAGS := --timestamp=none
endif

# The driver's revision lives in DRIVER_VERSION, independent of the app version: app-only
# releases ship the same driver, so updating the app never needs a driver reinstall. It is
# compiled in (served as the 'otvr' property) and stamped into Info.plist (OtoDriverVersion,
# CFBundleVersion, CFBundleShortVersionString). The Info.plist is regenerated on every run and
# replaced (forcing a re-sign) only when its content differs: make 3.81's one-second timestamps
# can't be trusted for this.
DRIVER_VERSION_FILE := DRIVER_VERSION
DRIVER_VERSION := $(strip $(shell cat $(DRIVER_VERSION_FILE) 2>/dev/null))
ifeq ($(shell printf '%s' '$(DRIVER_VERSION)' | grep -Eqx '[1-9][0-9]{0,8}' && echo ok),)
$(error DRIVER_VERSION must hold a positive integer, got '$(DRIVER_VERSION)')
endif
VFLAGS   := -DOTO_DRIVER_VERSION=$(DRIVER_VERSION)
PLIST    := $(BUNDLE)/Contents/Info.plist

all: sign $(VERIFY) sign-helper

$(PLIST): Info.plist $(DRIVER_VERSION_FILE) FORCE
	@mkdir -p $(BUNDLE)/Contents
	@cp Info.plist $@.tmp
	@plutil -replace OtoDriverVersion -integer '$(DRIVER_VERSION)' $@.tmp
	@plutil -replace CFBundleShortVersionString -string '$(DRIVER_VERSION)' $@.tmp
	@plutil -replace CFBundleVersion -string '$(DRIVER_VERSION)' $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else \
		echo "Info.plist: driver revision $(DRIVER_VERSION)"; mv $@.tmp $@; rm -f $(STAMP); fi

$(BIN): src/OtoDriver.c $(DRIVER_VERSION_FILE)
	@mkdir -p $(BUNDLE)/Contents/MacOS
	@rm -f $(STAMP)
	clang $(CFLAGS) $(VFLAGS) -bundle -fvisibility=hidden -o $@ src/OtoDriver.c \
		-framework CoreFoundation -framework CoreAudio

# Signs when the binary was rebuilt or the identity/flags differ from the last signing, which
# $(STAMP) records. No fallback: if SIGN_ID is not in the keychain, codesign fails and so does
# the build.
sign: $(BIN) $(PLIST)
	@if [ "$$(id -u)" = 0 ]; then echo "error: do not build the driver as root" >&2; exit 1; fi
	@cfg='$(SIGN_ID) $(SIGN_FLAGS)'; \
	if [ -f $(STAMP) ] && [ "$$(cat $(STAMP))" = "$$cfg" ]; then exit 0; fi; \
	rm -f $(STAMP); \
	if [ '$(SIGN_ID)' = - ]; then echo "signing ad-hoc"; else echo "signing with $(SIGN_ID)"; fi; \
	codesign --force $(SIGN_FLAGS) -s '$(SIGN_ID)' $(BUNDLE) && \
	codesign --verify --strict --verbose=1 $(BUNDLE) && \
	printf '%s\n' "$$cfg" > $(STAMP)

$(VERIFY): tools/oto-verify.c $(DRIVER_VERSION_FILE)
	@mkdir -p $(BUILD)
	clang $(CFLAGS) $(VFLAGS) -o $@ $< -framework CoreFoundation -framework CoreAudio

$(HELPER): tools/oto-hal-helper.c
	@mkdir -p $(BUILD)
	@rm -f $(HSTAMP)
	clang $(CFLAGS) -o $@ $< -framework CoreFoundation -framework CoreAudio

# Signed like the driver (same identity and flags) under its own identifier, which the install
# scripts and the app pin when they verify the root-owned copy before running it.
sign-helper: $(HELPER)
	@if [ "$$(id -u)" = 0 ]; then echo "error: do not build the helper as root" >&2; exit 1; fi
	@cfg='$(SIGN_ID) $(SIGN_FLAGS)'; \
	if [ -f $(HSTAMP) ] && [ "$$(cat $(HSTAMP))" = "$$cfg" ]; then exit 0; fi; \
	rm -f $(HSTAMP); \
	codesign --force $(SIGN_FLAGS) -i dev.oto.hal-helper -s '$(SIGN_ID)' $(HELPER) && \
	codesign --verify --strict --verbose=1 $(HELPER) && \
	printf '%s\n' "$$cfg" > $(HSTAMP)

test: $(HELPER)
	tools/test-hal-helper.sh $(HELPER)

# A distributable driver: refuses ad-hoc, unknown and non-Developer-ID identities.
release:
	@if [ '$(SIGN_ID)' = - ]; then \
		echo "error: make release needs SIGN_ID set to a Developer ID Application identity" >&2; exit 1; fi
	@security find-identity -v -p codesigning | grep -F -- '$(SIGN_ID)' | grep -q 'Developer ID Application' || { \
		echo "error: SIGN_ID '$(SIGN_ID)' is not a Developer ID Application identity in the keychain" >&2; exit 1; }
	@$(MAKE) --no-print-directory all RELEASE=1
	codesign --verify --strict -R='anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.13] exists' $(BUNDLE)
	codesign --verify --strict -R='anchor apple generic and identifier "dev.oto.hal-helper" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists' $(HELPER)

clean:
	rm -rf $(BUILD)

.PHONY: all sign sign-helper release test clean FORCE
FORCE:
