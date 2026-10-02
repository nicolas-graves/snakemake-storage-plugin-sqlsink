"""Credentials record loading: validation, secrecy of error messages, file
permissions, SOPS decryption (mocked and real), memoisation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from sqlsink import credentials as cred
from sqlsink.credentials import (
    CredentialsError,
    PgCredentials,
    load_credentials,
    resolve_url,
)

PW = "s3cr3t-pw"
GOOD = {"host": "db.example", "dbname": "app", "user": "alice", "password": PW}


@pytest.fixture(autouse=True)
def _clear_cache():
    load_credentials.cache_clear()
    yield
    load_credentials.cache_clear()


def _write(path, text, mode=0o600):
    path.write_text(text)
    path.chmod(mode)
    return path


def _chain(exc):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


def test_from_mapping_errors_hide_values():
    with pytest.raises(CredentialsError, match="dbname"):
        PgCredentials.from_mapping({k: v for k, v in GOOD.items() if k != "dbname"})
    with pytest.raises(CredentialsError, match="bogus") as ei:
        PgCredentials.from_mapping({**GOOD, "bogus": 1})
    assert PW not in str(ei.value)
    with pytest.raises(CredentialsError, match="port") as ei:
        PgCredentials.from_mapping({**GOOD, "port": "abc"}, source="x.yaml")
    assert PW not in str(ei.value) and "x.yaml" in str(ei.value)
    with pytest.raises(CredentialsError, match="password") as ei:
        PgCredentials.from_mapping({**GOOD, "password": 12345})
    assert "12345" not in str(ei.value)
    with pytest.raises(CredentialsError):
        PgCredentials.from_mapping(["not", "a", "mapping"])


def test_repr_url_round_trip():
    c = PgCredentials.from_mapping(
        {**GOOD, "port": "6543", "sslmode": "verify-full", "sslrootcert": "/ca.pem"}
    )
    assert PW not in repr(c)
    u = c.url()
    assert (u.drivername, u.username, u.password) == ("postgresql+psycopg", "alice", PW)
    assert (u.host, u.port, u.database) == ("db.example", 6543, "app")
    assert dict(u.query) == {"sslmode": "verify-full", "sslrootcert": "/ca.pem"}
    assert PW not in u.render_as_string(hide_password=True)
    assert dict(PgCredentials.from_mapping(GOOD).url().query) == {}
    assert PgCredentials.from_mapping(GOOD).port == 5432


def test_plaintext_permissions(tmp_path):
    p = _write(tmp_path / "c.yaml", json.dumps(GOOD), 0o644)
    with pytest.raises(CredentialsError, match="chmod 600"):
        load_credentials(p)
    p.chmod(0o600)
    assert load_credentials(p).password == PW


def test_malformed_record_does_not_leak_password(tmp_path):
    p = _write(tmp_path / "c.yaml", f'host: h\npassword: "unterminated {PW}\n')
    with pytest.raises(CredentialsError, match="malformed credentials record at line") as ei:
        load_credentials(p)
    for e in _chain(ei.value):
        assert PW not in str(e) and PW not in repr(e)
    assert ei.value.__context__ is None and ei.value.__cause__ is None


def test_missing_file(tmp_path):
    with pytest.raises(CredentialsError, match="nope.yaml"):
        load_credentials(tmp_path / "nope.yaml")


def _sops_file(tmp_path):
    return _write(tmp_path / "db.sops.yaml", "sops:\n  version: 3.9.0\nopaque: x\n", 0o644)


def test_sops_decrypt_and_memoise(tmp_path, monkeypatch):
    p = _sops_file(tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout=json.dumps(GOOD), stderr="")

    monkeypatch.setattr(cred.subprocess, "run", fake_run)
    assert load_credentials(p).password == PW
    assert load_credentials(p).user == "alice"
    assert len(calls) == 1
    assert calls[0][0] == ["sops", "--decrypt", "--output-type", "json", str(p)]
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    load_credentials(p)
    assert len(calls) == 2


def test_sops_failure(tmp_path, monkeypatch):
    p = _sops_file(tmp_path)
    monkeypatch.setattr(
        cred.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=128, stdout="", stderr="no key \n"),
    )
    with pytest.raises(CredentialsError, match=r"no key.*sops -d"):
        load_credentials(p)


def test_sops_not_installed(tmp_path, monkeypatch):
    p = _sops_file(tmp_path)

    def boom(*a, **k):
        raise FileNotFoundError("sops")

    monkeypatch.setattr(cred.subprocess, "run", boom)
    with pytest.raises(CredentialsError, match="not installed"):
        load_credentials(p)


@pytest.mark.skipif(
    not (shutil.which("sops") and shutil.which("age-keygen")),
    reason="needs sops and age-keygen",
)
def test_real_sops_round_trip(tmp_path, monkeypatch):
    key = tmp_path / "key.txt"
    out = subprocess.run(
        ["age-keygen", "-o", str(key)], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    pub = next(
        line.split()[-1]
        for line in (out.stdout + out.stderr + key.read_text()).splitlines()
        if "public key:" in line
    )
    plain = _write(tmp_path / "plain.yaml", json.dumps({**GOOD, "port": 6543}))
    enc = tmp_path / "db.sops.yaml"
    subprocess.run(
        ["sops", "--encrypt", "--age", pub, "--input-type", "yaml",
         "--output-type", "yaml", "--output", str(enc), str(plain)],
        check=True, cwd=tmp_path, capture_output=True,
    )
    assert PW not in enc.read_text()
    monkeypatch.setenv("SOPS_AGE_KEY_FILE", str(key))
    c = load_credentials(enc)
    assert c == PgCredentials.from_mapping({**GOOD, "port": 6543})


def test_resolve_url(tmp_path):
    p = _write(tmp_path / "c.yaml", json.dumps(GOOD))
    with pytest.raises(CredentialsError):
        resolve_url()
    with pytest.raises(CredentialsError):
        resolve_url(credentials=p, dsn="postgresql://x")
    assert resolve_url(dsn="postgresql://x/y") == "postgresql://x/y"
    assert resolve_url(credentials=p).host == "db.example"
