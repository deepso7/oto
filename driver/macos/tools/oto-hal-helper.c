// SPDX-License-Identifier: MIT OR Apache-2.0
//
// oto-hal-helper: installs and removes OtoMicrophone.driver in the HAL plug-in directory, one
// caller at a time, without the destination ever going missing mid-upgrade.
//
//   oto-hal-helper install <stage_driver> <hal_dir> <driver_name> <invoking_uid> [options]
//   oto-hal-helper uninstall <hal_dir> <driver_name> <invoking_uid> [options]
//   oto-hal-helper --check-audio-idle
//
// install / uninstall (run as root by the app's privileged step and scripts/macos/*-driver.sh):
//   1. Take an exclusive kernel flock() on <hal_dir>/.<driver_name>.lock (0600, never unlinked;
//      the kernel drops it on exit, crash or SIGKILL, so there is no stale-lock timeout).
//   2. install: copy <stage_driver> to <hal_dir>/.<driver_name>.swap-<nonce>, make it root-owned
//      and not group/world writable, verify its signature, then
//        - replace an existing driver with renamex_np(swap, dest, RENAME_SWAP), so dest is never
//          absent, and delete the swapped-out old copy, or
//        - rename it into place on a fresh install.
//      Refuses to replace a newer OtoDriverVersion unless --allow-downgrade.
//      uninstall: rename dest to .<driver_name>.removing-<nonce> and delete it.
//   3. Only after success, delete stale leftovers: .swap-*, .removing-*, the old installer's
//      .staging-* and the legacy .old-* (which is ambiguous between a failed install and a failed
//      uninstall, so it is never restored).
//   4. Ask, as <invoking_uid>, whether any audio device is running (this binary's
//      --check-audio-idle through launchctl asuser + sudo -u, killed after 3 s). Only if every
//      device is idle is coreaudiod restarted. Prints one of
//        OTO_RESULT=restarted   coreaudiod was restarted and loads the new state
//        OTO_RESULT=reboot      audio busy, check failed or restart failed: finish on restart
//        OTO_RESULT=absent      uninstall: nothing was installed
//        OTO_RESULT=installed / OTO_RESULT=removed   with --no-restart
//
//   Options:
//     --no-restart       skip step 4 (and the root requirement): lets tests drive the lock,
//                        naming and swap logic in a scratch directory as an ordinary user
//     --no-verify        skip the codesign check of the staged copy (tests only)
//     --allow-downgrade  install even if the installed driver reports a newer OtoDriverVersion
//
// --check-audio-idle: exits 0 iff every device in kAudioHardwarePropertyDevices reports
//   kAudioDevicePropertyDeviceIsRunningSomewhere = 0, 1 if any device is running, 2 on any Core
//   Audio error.
//
// Exit status of install / uninstall: 0 on success (whatever OTO_RESULT says), 1 on failure with
// nothing changed, 2 on bad usage, 3 when a downgrade was refused.

#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <copyfile.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fts.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <removefile.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/attr.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <time.h>
#include <unistd.h>

extern char** environ;

#define CHECK_TIMEOUT_MS 3000

static bool gNoRestart = false;
static bool gNoVerify = false;
static bool gAllowDowngrade = false;

static void die_usage(void) {
    fprintf(stderr,
            "usage: oto-hal-helper install <stage_driver> <hal_dir> <driver_name> <invoking_uid> [options]\n"
            "       oto-hal-helper uninstall <hal_dir> <driver_name> <invoking_uid> [options]\n"
            "       oto-hal-helper --check-audio-idle\n"
            "options: --no-restart --no-verify --allow-downgrade\n");
    exit(2);
}

// ---------------------------------------------------------------------------------------------
// --check-audio-idle

static int check_audio_idle(void) {
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    UInt32 size = 0;
    OSStatus st = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &a, 0, NULL, &size);
    if (st != noErr) {
        fprintf(stderr, "oto-hal-helper: listing devices failed (%d)\n", (int)st);
        return 2;
    }
    UInt32 n = size / sizeof(AudioObjectID);
    AudioObjectID* ids = calloc(n ? n : 1, sizeof *ids);
    if (ids == NULL) return 2;
    st = AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &size, ids);
    if (st != noErr) {
        fprintf(stderr, "oto-hal-helper: listing devices failed (%d)\n", (int)st);
        free(ids);
        return 2;
    }
    n = size / sizeof(AudioObjectID);
    int result = 0;
    AudioObjectPropertyAddress r = {kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    for (UInt32 i = 0; i < n; i++) {
        UInt32 running = 0, rs = sizeof running;
        st = AudioObjectGetPropertyData(ids[i], &r, 0, NULL, &rs, &running);
        if (st != noErr) {
            fprintf(stderr, "oto-hal-helper: device %u: IsRunningSomewhere failed (%d)\n", ids[i], (int)st);
            result = 2;
            break;
        }
        if (running) {
            fprintf(stderr, "oto-hal-helper: device %u is running\n", ids[i]);
            result = 1;
        }
    }
    free(ids);
    return result;
}

// ---------------------------------------------------------------------------------------------
// Paths and naming

static bool valid_driver_name(const char* name) {
    size_t n = strlen(name);
    return n > 7 && n < 200 && name[0] != '.' && strchr(name, '/') == NULL &&
           strcmp(name + n - 7, ".driver") == 0;
}

static void make_nonce(char* out, size_t n) {
    snprintf(out, n, "%d-%08x", (int)getpid(), arc4random());
}

static bool path_exists(const char* p) {
    struct stat st;
    return lstat(p, &st) == 0;
}

static bool is_real_dir(const char* p) {
    struct stat st;
    return lstat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static int remove_tree(const char* p) {
    removefile_state_t s = removefile_state_alloc();
    int r = removefile(p, s, REMOVEFILE_RECURSIVE);
    removefile_state_free(s);
    return r;
}

// Deletes every .<name>.<prefix>-* entry in hal_dir. Only called while holding the lock and after
// the requested operation succeeded, so none of them belongs to a live operation.
static void cleanup_leftovers(const char* hal_dir, const char* name) {
    static const char* const prefixes[] = {"swap-", "removing-", "staging-", "old-"};
    DIR* d = opendir(hal_dir);
    if (d == NULL) return;
    char base[PATH_MAX];
    snprintf(base, sizeof base, ".%s.", name);
    size_t baselen = strlen(base);
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, base, baselen) != 0) continue;
        const char* rest = e->d_name + baselen;
        for (size_t i = 0; i < sizeof prefixes / sizeof *prefixes; i++) {
            if (strncmp(rest, prefixes[i], strlen(prefixes[i])) != 0) continue;
            char p[PATH_MAX];
            snprintf(p, sizeof p, "%s/%s", hal_dir, e->d_name);
            if (remove_tree(p) == 0) printf("removed leftover %s\n", p);
            else fprintf(stderr, "warning: couldn't remove %s: %s\n", p, strerror(errno));
            break;
        }
    }
    closedir(d);
}

// ---------------------------------------------------------------------------------------------
// Lock

static int take_lock(const char* hal_dir, const char* name) {
    char p[PATH_MAX];
    snprintf(p, sizeof p, "%s/.%s.lock", hal_dir, name);
    int fd = open(p, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
        fprintf(stderr, "error: couldn't open lock %s: %s\n", p, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid()) {
        fprintf(stderr, "error: lock %s is not a regular file owned by uid %d\n", p, (int)geteuid());
        close(fd);
        return -1;
    }
    if ((st.st_mode & 07777) != 0600) fchmod(fd, 0600);
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno != EWOULDBLOCK) {
            fprintf(stderr, "error: flock %s: %s\n", p, strerror(errno));
            close(fd);
            return -1;
        }
        fprintf(stderr, "waiting for another driver install or uninstall to finish\n");
        if (flock(fd, LOCK_EX) != 0) {
            fprintf(stderr, "error: flock %s: %s\n", p, strerror(errno));
            close(fd);
            return -1;
        }
    }
    return fd; // held until exit
}

// ---------------------------------------------------------------------------------------------
// Driver revision on disk: Contents/Info.plist OtoDriverVersion (integer). -1 if unreadable;
// a valid plist without the key (drivers before DRIVER_VERSION existed) is revision 0.

static long disk_version(const char* bundle) {
    char p[PATH_MAX];
    snprintf(p, sizeof p, "%s/Contents/Info.plist", bundle);
    int fd = open(p, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    CFMutableDataRef data = CFDataCreateMutable(NULL, 0);
    UInt8 buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof buf)) > 0) CFDataAppendBytes(data, buf, n);
    close(fd);
    if (n < 0) {
        CFRelease(data);
        return -1;
    }
    CFPropertyListRef pl = CFPropertyListCreateWithData(NULL, data, kCFPropertyListImmutable, NULL, NULL);
    CFRelease(data);
    if (pl == NULL) return -1;
    long v = -1;
    if (CFGetTypeID(pl) == CFDictionaryGetTypeID()) {
        CFTypeRef x = CFDictionaryGetValue(pl, CFSTR("OtoDriverVersion"));
        if (x == NULL) v = 0;
        else if (CFGetTypeID(x) == CFNumberGetTypeID()) {
            long long ll = 0;
            if (CFNumberGetValue(x, kCFNumberLongLongType, &ll) && ll >= 0 && ll <= UINT32_MAX) v = (long)ll;
        }
    }
    CFRelease(pl);
    return v;
}

// ---------------------------------------------------------------------------------------------
// Child processes

// Spawns argv and waits up to timeout_ms (0 = forever). Returns the exit status, or -1 if it
// could not be spawned, timed out (then killed) or died on a signal.
static int run(char* const argv[], int timeout_ms) {
    pid_t pid;
    int err = posix_spawn(&pid, argv[0], NULL, NULL, argv, environ);
    if (err != 0) {
        fprintf(stderr, "error: couldn't run %s: %s\n", argv[0], strerror(err));
        return -1;
    }
    int status = 0;
    if (timeout_ms <= 0) {
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    } else {
        int waited = 0;
        for (;;) {
            pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) break;
            if (r < 0 && errno != EINTR) return -1;
            if (waited >= timeout_ms) {
                fprintf(stderr, "%s timed out after %d ms\n", argv[0], timeout_ms);
                kill(pid, SIGKILL);
                while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
                return -1;
            }
            struct timespec ts = {0, 20 * 1000 * 1000};
            nanosleep(&ts, NULL);
            waited += 20;
        }
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static bool self_path(char* out, size_t n) {
    char raw[PATH_MAX];
    uint32_t size = sizeof raw;
    if (_NSGetExecutablePath(raw, &size) != 0) return false;
    char resolved[PATH_MAX];
    if (realpath(raw, resolved) == NULL) return false;
    snprintf(out, n, "%s", resolved);
    return true;
}

// Asks, as the invoking user (whose session sees what apps are playing or recording), whether
// all audio is idle, and restarts coreaudiod only then. Always prints OTO_RESULT.
static void restart_if_idle(uid_t uid) {
    char self[PATH_MAX];
    int idle = -1;
    if (!self_path(self, sizeof self)) {
        fprintf(stderr, "couldn't locate this executable; not restarting coreaudiod\n");
    } else if (uid == 0) {
        char* argv[] = {self, "--check-audio-idle", NULL};
        idle = run(argv, CHECK_TIMEOUT_MS);
    } else {
        char uidarg[32], sudouid[32];
        snprintf(uidarg, sizeof uidarg, "%u", (unsigned)uid);
        snprintf(sudouid, sizeof sudouid, "#%u", (unsigned)uid);
        char* argv[] = {"/bin/launchctl", "asuser", uidarg, "/usr/bin/sudo", "-n", "-u", sudouid,
                        self, "--check-audio-idle", NULL};
        idle = run(argv, CHECK_TIMEOUT_MS);
    }
    if (idle != 0) {
        printf(idle == 1 ? "Audio is playing or recording; not restarting coreaudiod.\n"
                         : "Couldn't confirm audio is idle; not restarting coreaudiod.\n");
        printf("OTO_RESULT=reboot\n");
        return;
    }
    // launchctl kickstart is refused under SIP on macOS 27; launchd relaunches coreaudiod.
    char* argv[] = {"/usr/bin/killall", "coreaudiod", NULL};
    if (run(argv, 0) == 0) {
        printf("OTO_RESULT=restarted\n");
    } else {
        printf("Couldn't restart coreaudiod.\n");
        printf("OTO_RESULT=reboot\n");
    }
}

// ---------------------------------------------------------------------------------------------
// Staging

// Root-owned, nothing group/world writable, no quarantine: what coreaudiod will load.
static int secure_tree(const char* root) {
    char* paths[] = {(char*)root, NULL};
    FTS* fts = fts_open(paths, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
    if (fts == NULL) return -1;
    bool asRoot = geteuid() == 0;
    int rc = 0;
    FTSENT* e;
    while ((e = fts_read(fts)) != NULL) {
        if (e->fts_info == FTS_DP) continue;
        if (e->fts_info == FTS_ERR || e->fts_info == FTS_DNR || e->fts_info == FTS_NS) {
            rc = -1;
            break;
        }
        if (asRoot && lchown(e->fts_accpath, 0, 0) != 0) { rc = -1; break; }
        if (e->fts_info != FTS_SL && e->fts_info != FTS_SLNONE) {
            mode_t m = e->fts_statp->st_mode & 07777 & ~(mode_t)(S_IWGRP | S_IWOTH);
            if (chmod(e->fts_accpath, m) != 0) { rc = -1; break; }
        }
        removexattr(e->fts_accpath, "com.apple.quarantine", XATTR_NOFOLLOW);
    }
    fts_close(fts);
    return rc;
}

static bool verify_signature(const char* bundle) {
    char* argv[] = {"/usr/bin/codesign", "--verify", "--strict", (char*)bundle, NULL};
    return run(argv, 0) == 0;
}

// ---------------------------------------------------------------------------------------------
// install / uninstall

static int do_install(const char* stage, const char* hal_dir, const char* name, uid_t uid) {
    if (!is_real_dir(stage)) {
        fprintf(stderr, "error: %s is not a directory\n", stage);
        return 1;
    }
    if (take_lock(hal_dir, name) < 0) return 1;

    char dest[PATH_MAX], swap[PATH_MAX], nonce[64];
    snprintf(dest, sizeof dest, "%s/%s", hal_dir, name);
    make_nonce(nonce, sizeof nonce);
    snprintf(swap, sizeof swap, "%s/.%s.swap-%s", hal_dir, name, nonce);

    long to = disk_version(stage);
    if (to < 0) {
        fprintf(stderr, "error: couldn't read %s/Contents/Info.plist\n", stage);
        return 1;
    }
    bool had = path_exists(dest);
    long from = had ? disk_version(dest) : -1;
    if (had) printf("OTO_FROM=%ld\n", from);
    else printf("OTO_FROM=absent\n");
    printf("OTO_TO=%ld\n", to);
    if (had && from > to && !gAllowDowngrade) {
        fprintf(stderr, "error: the installed driver (revision %ld) is newer than this one (%ld)\n", from, to);
        return 3;
    }

    if (copyfile(stage, swap, NULL, COPYFILE_ALL | COPYFILE_RECURSIVE | COPYFILE_NOFOLLOW) != 0) {
        fprintf(stderr, "error: couldn't copy %s to %s: %s\n", stage, swap, strerror(errno));
        remove_tree(swap);
        return 1;
    }
    if (secure_tree(swap) != 0) {
        fprintf(stderr, "error: couldn't set ownership/permissions on %s\n", swap);
        remove_tree(swap);
        return 1;
    }
    if (!gNoVerify && !verify_signature(swap)) {
        fprintf(stderr, "error: the staged copy failed signature verification\n");
        remove_tree(swap);
        return 1;
    }
    // Re-read after copying: what is installed is what was verified.
    if (disk_version(swap) != to) {
        fprintf(stderr, "error: the staged copy's revision changed while copying\n");
        remove_tree(swap);
        return 1;
    }

    if (had) {
        // One atomic step: dest becomes the new driver, swap holds the old one.
        if (renamex_np(swap, dest, RENAME_SWAP) != 0) {
            fprintf(stderr, "error: couldn't swap in the new driver: %s\n", strerror(errno));
            remove_tree(swap);
            return 1;
        }
        if (remove_tree(swap) != 0) fprintf(stderr, "warning: couldn't remove the old driver at %s\n", swap);
    } else {
        int r = renamex_np(swap, dest, RENAME_EXCL);
        if (r != 0 && (errno == ENOTSUP || errno == EINVAL)) r = rename(swap, dest);
        if (r != 0) {
            fprintf(stderr, "error: couldn't move the new driver into place: %s\n", strerror(errno));
            remove_tree(swap);
            return 1;
        }
    }
    printf("Installed %s (revision %ld)\n", dest, to);
    cleanup_leftovers(hal_dir, name);

    if (gNoRestart) printf("OTO_RESULT=installed\n");
    else restart_if_idle(uid);
    return 0;
}

static int do_uninstall(const char* hal_dir, const char* name, uid_t uid) {
    if (take_lock(hal_dir, name) < 0) return 1;
    char dest[PATH_MAX], gone[PATH_MAX], nonce[64];
    snprintf(dest, sizeof dest, "%s/%s", hal_dir, name);
    if (!path_exists(dest)) {
        cleanup_leftovers(hal_dir, name);
        printf("OTO_RESULT=absent\n");
        return 0;
    }
    make_nonce(nonce, sizeof nonce);
    snprintf(gone, sizeof gone, "%s/.%s.removing-%s", hal_dir, name, nonce);
    // Rename first: the driver leaves its place in one step, whatever happens to the delete.
    if (rename(dest, gone) != 0) {
        fprintf(stderr, "error: couldn't remove %s: %s\n", dest, strerror(errno));
        return 1;
    }
    if (remove_tree(gone) != 0) fprintf(stderr, "warning: couldn't delete %s\n", gone);
    printf("Removed %s\n", dest);
    cleanup_leftovers(hal_dir, name);

    if (gNoRestart) printf("OTO_RESULT=removed\n");
    else restart_if_idle(uid);
    return 0;
}

static uid_t parse_uid(const char* s) {
    char* end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || end == s || *end || v > UINT32_MAX - 1) die_usage();
    return (uid_t)v;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc == 2 && strcmp(argv[1], "--check-audio-idle") == 0) return check_audio_idle();

    // Options may appear anywhere after the command; collect positionals separately.
    const char* pos[8];
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-restart") == 0) gNoRestart = true;
        else if (strcmp(argv[i], "--no-verify") == 0) gNoVerify = true;
        else if (strcmp(argv[i], "--allow-downgrade") == 0) gAllowDowngrade = true;
        else if (argv[i][0] == '-' && argv[i][1] == '-') die_usage();
        else if (npos < 8) pos[npos++] = argv[i];
        else die_usage();
    }
    if (npos < 1) die_usage();

    bool install = strcmp(pos[0], "install") == 0;
    bool uninstall = strcmp(pos[0], "uninstall") == 0;
    if (!(install && npos == 5) && !(uninstall && npos == 4)) die_usage();
    const char* hal_dir = install ? pos[2] : pos[1];
    const char* name = install ? pos[3] : pos[2];
    uid_t uid = parse_uid(install ? pos[4] : pos[3]);

    if (!valid_driver_name(name)) {
        fprintf(stderr, "error: invalid driver name '%s'\n", name);
        return 2;
    }
    if (!is_real_dir(hal_dir)) {
        fprintf(stderr, "error: %s is not a directory\n", hal_dir);
        return 1;
    }
    if (!gNoRestart && geteuid() != 0) {
        fprintf(stderr, "error: restarting coreaudiod needs root (or pass --no-restart)\n");
        return 2;
    }
    umask(022);
    return install ? do_install(pos[1], hal_dir, name, uid) : do_uninstall(hal_dir, name, uid);
}
