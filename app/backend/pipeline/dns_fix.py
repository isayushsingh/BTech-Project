"""This network's default DNS resolver returns a bogus IP for
api.themoviedb.org (and themoviedb.org generally) that just hangs instead of
connecting -- confirmed by comparing against 1.1.1.1/8.8.8.8, which resolve
it correctly. Rather than requiring a system-wide VPN/DNS change, resolve
hostnames ourselves via a public DNS server and patch socket.getaddrinfo so
`requests` connects to the right IP. SNI/Host headers are unaffected since
those use the original hostname string, not the resolved address.

Import this before making any requests to a domain affected by the block.
"""
import socket

import dns.resolver

_resolver = dns.resolver.Resolver(configure=False)
_resolver.nameservers = ["1.1.1.1", "8.8.8.8"]
_resolver.timeout = 3
_resolver.lifetime = 3

_original_getaddrinfo = socket.getaddrinfo
_cache: dict[str, str] = {}


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    ip = _cache.get(host)
    if ip is None:
        try:
            answer = _resolver.resolve(host, "A")
            ip = answer[0].to_text()
            _cache[host] = ip
        except Exception:
            return _original_getaddrinfo(host, port, family, type, proto, flags)
    return _original_getaddrinfo(ip, port, family, type, proto, flags)


socket.getaddrinfo = _patched_getaddrinfo
