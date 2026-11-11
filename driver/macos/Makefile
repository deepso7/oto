# Builds OtoMicrophone.driver (universal arm64 + x86_64) and the oto-verify tool into build/.
#
#   make                          build and sign ad-hoc (local development)
#   SIGN_ID="<identity>" make     build and sign with that keychain identity
#   SIGN_ID="Developer ID Application: Name (TEAMID)" make release
#                                 build for distribution: hardened runtime + secure timestamp;
#                                 fails unless SIGN_ID is a Developer ID identity in the keychain
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

# The driver carries the workspace version (Cargo.toml), so a driver and the app that ships it
# always agree. The Info.plist is regenerated on every run and replaced (forcing a re-sign) only
# when its content differs: make 3.81's one-second timestamps can't be trusted for this.
VERSION  ?= $(shell awk '/^\[workspace\.package\]/ { p = 1; next } /^\[/ { p = 0 } \
                p && $$1 == "version" { gsub(/"/, "", $$3); print $$3; exit }' ../../Cargo.toml)
ifeq ($(strip $(VERSION)),)
$(error couldn't read the workspace version from ../../Cargo.toml)
endif
PLIST    := $(BUNDLE)/Contents/Info.plist

all: sign $(VERIFY)

$(PLIST): FORCE
	@mkdir -p $(BUNDLE)/Contents
	@cp Info.plist $@.tmp
	@plutil -replace CFBundleShortVersionString -string '$(VERSION)' $@.tmp
	@plutil -replace CFBundleVersion -string '$(VERSION)' $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else \
		echo "Info.plist: version $(VERSION)"; mv $@.tmp $@; rm -f $(STAMP); fi

$(BIN): src/OtoDriver.c
	@mkdir -p $(BUNDLE)/Contents/MacOS
	@rm -f $(STAMP)
	clang $(CFLAGS) -bundle -fvisibility=hidden -o $@ src/OtoDriver.c \
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

$(VERIFY): tools/oto-verify.c
	@mkdir -p $(BUILD)
	clang $(CFLAGS) -o $@ $< -framework CoreFoundation -framework CoreAudio

# A distributable driver: refuses ad-hoc, unknown and non-Developer-ID identities.
release:
	@if [ '$(SIGN_ID)' = - ]; then \
		echo "error: make release needs SIGN_ID set to a Developer ID Application identity" >&2; exit 1; fi
	@security find-identity -v -p codesigning | grep -F -- '$(SIGN_ID)' | grep -q 'Developer ID Application' || { \
		echo "error: SIGN_ID '$(SIGN_ID)' is not a Developer ID Application identity in the keychain" >&2; exit 1; }
	@$(MAKE) --no-print-directory all RELEASE=1
	codesign --verify --strict -R='anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.13] exists' $(BUNDLE)

clean:
	rm -rf $(BUILD)

.PHONY: all sign release clean FORCE
FORCE:
