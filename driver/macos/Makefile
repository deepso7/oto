# Builds OtoMicrophone.driver (universal arm64 + x86_64) and the oto-verify tool into build/.
#   make                     build, then sign with $(SIGN_ID) if present in the keychain, else ad-hoc
#   make SIGN_ID=-           force ad-hoc signing
#   make clean
# Install with scripts/macos/install-driver.sh (it runs make first).

BUILD    := build
BUNDLE   := $(BUILD)/OtoMicrophone.driver
BIN      := $(BUNDLE)/Contents/MacOS/OtoMicrophone
STAMP    := $(BUILD)/.signed
VERIFY   := $(BUILD)/oto-verify
ARCHS    := -arch arm64 -arch x86_64
MINOS    := -mmacosx-version-min=12.0
CFLAGS   := -std=c11 -O2 -Wall -Wextra -Werror $(ARCHS) $(MINOS)
SIGN_ID  ?= Apple Development: deeps.o7o8.y@gmail.com (RH3RCAWYF5)

all: $(STAMP) $(VERIFY)

$(BIN): src/OtoDriver.c Info.plist
	@mkdir -p $(BUNDLE)/Contents/MacOS
	cp Info.plist $(BUNDLE)/Contents/Info.plist
	clang $(CFLAGS) -bundle -fvisibility=hidden -o $@ src/OtoDriver.c \
		-framework CoreFoundation -framework CoreAudio

$(STAMP): $(BIN)
	@if [ "$(SIGN_ID)" != "-" ] && security find-identity -v -p codesigning | grep -qF "$(SIGN_ID)"; then \
		echo "signing with $(SIGN_ID)"; \
		codesign --force --timestamp=none -s "$(SIGN_ID)" $(BUNDLE); \
	else \
		echo "signing ad-hoc"; \
		codesign --force -s - $(BUNDLE); \
	fi
	codesign --verify --strict --verbose=1 $(BUNDLE)
	@touch $@

$(VERIFY): tools/oto-verify.c
	@mkdir -p $(BUILD)
	clang $(CFLAGS) -o $@ $< -framework CoreFoundation -framework CoreAudio

clean:
	rm -rf $(BUILD)

.PHONY: all clean
