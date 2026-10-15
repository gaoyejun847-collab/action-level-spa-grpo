"""Release freed glibc allocations after temporary CPU checkpoint copies."""
import ctypes
import gc
import os
import sys


def release_cpu_cache(label):
    def rss():
        if sys.platform.startswith('linux'):
            with open('/proc/self/statm') as f:
                return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20
        return 0
    before = rss()
    gc.collect()
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None)
        if hasattr(libc, 'malloc_trim'):
            libc.malloc_trim(0)
    print(f'CPU_CHECKPOINT_MEMORY {label} pid={os.getpid()} rss_mib={before:.1f}->{rss():.1f}', flush=True)
