# SPDX-License-Identifier: MIT
# Copyright 2026 (c) Mitja Goroshevsky and GOSH Technology Ltd.
#
# Self-signed TLS cert generation + join token for gosh.memory.

import base64
import datetime
import hashlib
import json
import ssl
from pathlib import Path


def ensure_cert(data_dir: str) -> tuple[str, str]:
    """Return (certfile, keyfile) paths. Generate self-signed if missing."""
    cert_dir = Path(data_dir) / ".tls"
    cert_dir.mkdir(parents=True, exist_ok=True)
    certfile = cert_dir / "cert.pem"
    keyfile = cert_dir / "key.pem"

    if certfile.exists() and keyfile.exists():
        return str(certfile), str(keyfile)

    _generate_self_signed(certfile, keyfile)
    certfile.chmod(0o600)
    keyfile.chmod(0o600)
    return str(certfile), str(keyfile)


def _generate_self_signed(certfile: Path, keyfile: Path):
    """Generate a self-signed certificate using cryptography library."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise RuntimeError(
            "TLS requires the 'cryptography' package.\n"
            "Install: pip install cryptography"
        )

    key = ec.generate_private_key(ec.SECP256R1())

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "gosh-memory"),
    ])

    import ipaddress
    import socket

    san_entries = [
        x509.DNSName("localhost"),
        x509.DNSName(socket.gethostname()),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=3650)
        )
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def cert_fingerprint(certfile: str) -> str:
    """SHA-256 fingerprint of a PEM certificate."""
    from cryptography import x509

    pem = Path(certfile).read_bytes()
    cert = x509.load_pem_x509_certificate(pem)
    digest = cert.fingerprint(cert.signature_hash_algorithm or __import__("cryptography.hazmat.primitives.hashes", fromlist=["SHA256"]).SHA256())
    return "sha256:" + hashlib.sha256(
        Path(certfile).read_bytes()
    ).hexdigest()




def make_join_token(url: str, token: str, certfile: str) -> str:
    """Create a portable join token for agents."""
    ca_pem = Path(certfile).read_text()
    fingerprint = cert_fingerprint(certfile)

    payload = json.dumps({
        "url": url,
        "token": token,
        "fingerprint": fingerprint,
        "ca": ca_pem,
    }, separators=(",", ":"))

    b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"gosh_join_{b64}"
