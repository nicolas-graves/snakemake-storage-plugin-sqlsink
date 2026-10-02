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


# --- references into a shared document -------------------------------------

PEM = "-----BEGIN CERTIFICATE-----\nMIIBdummy\n-----END CERTIFICATE-----\n"
SECTION = "superset-analytics-database"
SHARED = {
    # not a valid record on its own: must be ignored by a fragment reference
    "francetravail": {"client_id": "x", "bogus": [1, 2]},
    "stray": "scalar",
    SECTION: {
        "host": "pg.example.net",
        "port": 20184,
        "database": "superset_analytics",
        "ca_path": "/run/secrets/analytics-database-ca.pem",
        "ca": PEM,
        "loader": {"username": "loader_role", "password": PW},
        "runtime": {"username": "runtime_role", "password": "runtime-pw"},
    },
}


def _shared(tmp_path, doc=None):
    return _write(tmp_path / "all.yaml", json.dumps(SHARED if doc is None else doc))


def test_fragment_merge(tmp_path):
    p = _shared(tmp_path)
    c = load_credentials(f"{p}#{SECTION}.loader")
    assert (c.host, c.port, c.dbname) == ("pg.example.net", 20184, "superset_analytics")
    assert (c.user, c.password) == ("loader_role", PW)
    rt = load_credentials(f"{p}#{SECTION}.runtime")
    assert (rt.user, rt.password) == ("runtime_role", "runtime-pw")


def test_aliases_and_conflict(tmp_path):
    p = _write(tmp_path / "a.yaml", json.dumps({"host": "h", "database": "d", "username": "u", "password": PW}))
    c = load_credentials(p)
    assert (c.dbname, c.user) == ("d", "u")
    p2 = _write(tmp_path / "b.yaml", json.dumps({**GOOD, "database": "other"}))
    with pytest.raises(CredentialsError, match="database.*dbname") as ei:
        load_credentials(p2)
    assert PW not in str(ei.value)


def test_missing_path_key(tmp_path):
    p = _shared(tmp_path)
    with pytest.raises(CredentialsError, match=rf"no key '{SECTION}\.nope'") as ei:
        load_credentials(f"{p}#{SECTION}.nope")
    assert PW not in str(ei.value)
    with pytest.raises(CredentialsError, match="no key 'absent'"):
        load_credentials(f"{p}#absent")


def test_inline_ca_materialised(tmp_path):
    p = _shared(tmp_path)
    c = load_credentials(f"{p}#{SECTION}.loader")
    ca = c.sslrootcert
    assert ca != "/run/secrets/analytics-database-ca.pem"
    assert open(ca).read() == PEM
    assert os.stat(ca).st_mode & 0o777 == 0o600
    assert os.stat(os.path.dirname(ca)).st_mode & 0o777 == 0o700
    assert c.sslmode == "verify-full"
    assert dict(c.url().query)["sslrootcert"] == ca
    # explicit sslmode is kept; ca_path alone also defaults to verify-full
    doc = {**GOOD, "ca_path": "/ca.pem"}
    assert load_credentials(_write(tmp_path / "x.yaml", json.dumps(doc))).sslmode == "verify-full"
    doc = {**GOOD, "ca": PEM, "sslmode": "verify-ca"}
    assert load_credentials(_write(tmp_path / "y.yaml", json.dumps(doc))).sslmode == "verify-ca"


def test_inline_ca_must_be_pem(tmp_path):
    p = _write(tmp_path / "c.yaml", json.dumps({**GOOD, "ca": "not a certificate"}))
    with pytest.raises(CredentialsError, match="PEM") as ei:
        load_credentials(p)
    assert "not a certificate" not in str(ei.value) and PW not in str(ei.value)


def test_query_overrides(tmp_path):
    p = _shared(tmp_path)
    c = load_credentials(f"{p}#{SECTION}.loader?hostaddr=127.0.0.1&port=15432")
    assert (c.host, c.hostaddr, c.port) == ("pg.example.net", "127.0.0.1", 15432)
    assert dict(c.url().query)["hostaddr"] == "127.0.0.1"
    # no fragment: the query follows the path directly
    q = _write(tmp_path / "q.yaml", json.dumps(GOOD))
    assert load_credentials(f"{q}?port=1&sslmode=require").sslmode == "require"
    for bad in ("password=x", "user=x", "username=x", "ca=x", "bogus=1", "port"):
        with pytest.raises(CredentialsError) as ei:
            load_credentials(f"{p}#{SECTION}.loader?{bad}")
        assert PW not in str(ei.value) and "=x" not in str(ei.value)


def _fake_sops(calls, doc=SHARED):
    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "--extract" in cmd:
            key = json.loads(cmd[cmd.index("--extract") + 1])[0]
            return SimpleNamespace(returncode=0, stdout=json.dumps(doc[key]), stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(doc), stderr="")

    return fake_run


def test_sops_extract_command_and_memoisation(tmp_path, monkeypatch):
    p = _sops_file(tmp_path)
    calls = []
    monkeypatch.setattr(cred.subprocess, "run", _fake_sops(calls))
    load_credentials(f"{p}#{SECTION}.loader")
    assert calls[0] == ["sops", "--decrypt", "--output-type", "json",
                        "--extract", f'["{SECTION}"]', str(p)]
    load_credentials(f"{p}#{SECTION}.runtime")
    assert len(calls) == 1  # same section: one sops call
    # no fragment: no --extract
    calls.clear()
    monkeypatch.setattr(cred.subprocess, "run", _fake_sops(calls, GOOD))
    load_credentials(p)
    assert "--extract" not in calls[0]


def test_sqlsink_sops_env(tmp_path, monkeypatch):
    p = _sops_file(tmp_path)
    calls = []
    monkeypatch.setattr(cred.subprocess, "run", _fake_sops(calls, GOOD))
    monkeypatch.setenv("SQLSINK_SOPS", "/opt/pinned/sops")
    load_credentials(p)
    assert calls[0][0] == "/opt/pinned/sops"


@pytest.mark.skipif(
    not (shutil.which("sops") and shutil.which("age-keygen")),
    reason="needs sops and age-keygen",
)
def test_real_sops_shared_document(tmp_path, monkeypatch):
    key = tmp_path / "key.txt"
    subprocess.run(["age-keygen", "-o", str(key)], capture_output=True, text=True, check=True)
    pub = next(l.split()[-1] for l in key.read_text().splitlines() if "public key:" in l)
    plain = _write(tmp_path / "plain.yaml", json.dumps(SHARED))
    enc = tmp_path / "all.sops.yaml"
    subprocess.run(
        ["sops", "--encrypt", "--age", pub, "--input-type", "yaml", "--output-type", "yaml",
         "--output", str(enc), str(plain)],
        check=True, cwd=tmp_path, capture_output=True,
    )
    monkeypatch.setenv("SOPS_AGE_KEY_FILE", str(key))
    c = load_credentials(f"{enc}#{SECTION}.loader")
    assert (c.host, c.port, c.dbname, c.user, c.password) == (
        "pg.example.net", 20184, "superset_analytics", "loader_role", PW)
    assert open(c.sslrootcert).read() == PEM
