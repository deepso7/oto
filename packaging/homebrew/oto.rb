# Reference cask for the deepso7/homebrew-oto tap (Casks/oto.rb). The release workflow renders
# @VERSION@ and @SHA256@ (the published DMG's checksum) and pushes it to the tap, only ever
# moving forward (scripts/macos/publish-release.sh). Edit this file, not the tap's copy.
cask "oto" do
  version "@VERSION@"
  sha256 "@SHA256@"

  url "https://github.com/deepso7/oto/releases/download/v#{version}/Oto-#{version}.dmg"
  name "Oto"
  desc "Local, real-time noise suppression for your microphone"
  homepage "https://github.com/deepso7/oto"

  livecheck do
    url :url
    strategy :github_latest
  end

  # Oto updates itself in place; `brew upgrade` leaves it alone unless run with --greedy.
  auto_updates true
  depends_on macos: ">= :monterey"

  app "Oto.app"

  # The Oto Microphone audio driver (/Library/Audio/Plug-Ins/HAL/OtoMicrophone.driver) is
  # machine-wide, may be shared by other users, and needs an audio-checked coreaudiod restart
  # to remove, so the cask never deletes it: Oto’s own Settings → Remove the driver does.
  uninstall quit: "dev.oto.app"

  zap trash: [
    "~/Library/Application Support/oto",
    "~/Library/Caches/dev.oto.app",
    "~/Library/LaunchAgents/dev.oto.app.plist",
    "~/Library/Logs/Oto",
    "~/Library/Preferences/dev.oto.app.plist",
  ]

  caveats <<~EOS
    Oto installs its audio driver (Oto Microphone) the first time you open it, and asks for
    your password to do so. `brew uninstall oto` does not remove the driver, since it is
    shared by every user on this Mac. To remove it, open Oto → Settings → "Remove the
    driver" (Uninstall…) before uninstalling the app.
  EOS
end
