"""PostgreSQL credentials loaded from a structured, SOPS-encrypted record.

A record is a YAML/JSON mapping with separate fields (host, dbname, user,
password, ...) instead of a raw DSN, so the password never travels inside a
URL string that gets logged or echoed. An encrypted record (top-level `sops`
key) is decrypted by shelling out to `sops` (`$SQLSINK_SOPS`, default `sops`);
a plaintext record is only accepted when it is private to its owner, like
~/.pgpass.

The `credentials` setting is a reference, `PATH[#key.subkey...][?name=value&...]`,
so a deployment can keep ALL its secrets in ONE shared document and point
sqlsink at one role inside it. Decrypted shape of such a document:

    francetravail: {...}                  # other consumers' secrets, ignored
    superset-analytics-database:
      host: pg.example.net
      port: 20184
      database: superset_analytics
      ca_path: /run/secrets/analytics-database-ca.pem   # another consumer's path
      ca: |                                             # inline PEM CA
        -----BEGIN CERTIFICATE-----
        ...
      loader:  {username: loader_role,  password: ...}
      runtime: {username: runtime_role, password: ...}

    credentials: secrets/all.sops.yaml#superset-analytics-database.loader
    credentials: secrets/all.sops.yaml#superset-analytics-database.loader?hostaddr=127.0.0.1&port=15432

- `PATH` alone: the whole document is the record.
- `#a.b`: dotted path of mapping keys (a key may contain `-`). The record is
  the merge of the NON-mapping fields of each level along the path, outer to
  inner, inner winning; mappings off the path (`runtime`) are ignored. Only
  the first key is decrypted (`sops --extract`).
- Aliases, applied after merging: database -> dbname, username -> user,
  ca_path -> sslrootcert; an alias and its canonical name together is an error.
- `ca` (inline PEM) is written once per process to a 0600 file in a private
  0700 directory removed at exit; it becomes `sslrootcert`. With a CA and no
  `sslmode`, `sslmode` defaults to verify-full.
- `?name=value&...` overrides NON-secret fields only (host, hostaddr, port,
  sslmode, sslrootcert, sslcert, sslkey, options, dbname), e.g. `hostaddr` to
  reach the server through an ssh tunnel while TLS still verifies `host`.
  user, password and ca are refused there.

The grammar is strictly `PATH#fragment?query` (`?` after the fragment, or
after PATH when there is none). A `#` or `?` inside a real file path is not
supported.

Error messages name the source and the offending KEYS, never any value.
This is a leaf module: stdlib, yaml and sqlalchemy only.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import parse_qsl

import yaml
from sqlalchemy.engine import URL

# Optional libpq connection options; passed to the driver as URL query keys.
_QUERY_KEYS = ("hostaddr", "sslmode", "sslrootcert", "sslcert", "sslkey", "options")
_ALIASES = {"database": "dbname", "username": "user", "ca_path": "sslrootcert"}
# Non-secret fields a `?name=value` override may set.
_OVERRIDABLE = ("host", "hostaddr", "port", "sslmode", "sslrootcert", "sslcert", "sslkey", "options", "dbname")
_SECRET_FIELDS = ("user", "username", "password", "ca")


class CredentialsError(ValueError):
    """The credentials record is missing, unreadable, or invalid."""


@dataclass(frozen=True, kw_only=True)
class PgCredentials:
    host: str
    dbname: str
    user: str
    password: str = field(repr=False)
    port: int = 5432
    sslmode: str | None = None
    sslrootcert: str | None = None
    sslcert: str | None = None
    sslkey: str | None = None
    options: str | None = None
    hostaddr: str | None = None

    @classmethod
    def from_mapping(cls, data, *, source: str = "credentials") -> PgCredentials:
        if not isinstance(data, Mapping):
            raise CredentialsError(f"{source}: credentials record must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(k) for k in data if k not in known)
        if unknown:
            raise CredentialsError(f"{source}: unknown keys: {', '.join(unknown)}")
        required = ("host", "dbname", "user", "password")
        missing = [k for k in required if k not in data]
        if missing:
            raise CredentialsError(f"{source}: missing keys: {', '.join(missing)}")
        values = dict(data)
        if "port" in values:
            port = values["port"]
            # bool is an int subclass; "5432" strings are accepted, True is not
            if isinstance(port, bool):
                raise CredentialsError(f"{source}: key 'port' must be an integer")
            try:
                values["port"] = int(port)
            except (TypeError, ValueError):
                raise CredentialsError(f"{source}: key 'port' must be an integer") from None
        for key, val in values.items():
            if key == "port":
                continue
            if val is None and key in _QUERY_KEYS:
                continue
            if not isinstance(val, str):
                raise CredentialsError(f"{source}: key '{key}' must be a string")
        return cls(**values)

    def url(self) -> URL:
        query = {k: getattr(self, k) for k in _QUERY_KEYS if getattr(self, k) is not None}
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.dbname,
            query=query,
        )


# One Snakemake process builds the DAG with one call per wildcard; decrypt
# each section once. Keyed on mtime so an edited/rotated record is picked up.
_cache: dict[tuple[str, int, str], PgCredentials] = {}
_data_cache: dict[tuple[str, int, str | None], object] = {}
_ca_files: dict[str, str] = {}
_ca_dir: list[str] = []


def _sops_bin() -> str:
    # A deployment may use a pinned sops binary that is not on PATH.
    return os.environ.get("SQLSINK_SOPS", "sops")


def _decrypt(path: Path, first_key: str | None) -> object:
    cmd = [_sops_bin(), "--decrypt", "--output-type", "json"]
    if first_key is not None:
        # Decrypt only the section we need into memory.
        cmd += ["--extract", f"[{json.dumps(first_key)}]"]
    try:
        proc = subprocess.run(cmd + [str(path)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise CredentialsError(
            f"{path}: record is SOPS-encrypted but sops is not installed"
        ) from None
    if proc.returncode != 0:
        raise CredentialsError(
            f"{path}: sops failed to decrypt: {proc.stderr.strip()} "
            f"(run `sops -d {path}` to check)"
        )
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        raise CredentialsError(f"{path}: sops produced output that is not valid JSON") from None
    # --extract returns the bare section; rewrap so both shapes look alike.
    return {first_key: out} if first_key is not None else out


def _split_reference(ref: str) -> tuple[str, list[str], str]:
    path, sep, rest = ref.partition("#")
    if sep:
        frag, _, query = rest.partition("?")
    else:
        path, _, query = ref.partition("?")
        frag = ""
    keys = frag.split(".") if frag else []
    if sep and (not frag or not all(keys)):
        raise CredentialsError(f"{path}: empty key in credentials reference '#{frag}'")
    return path, keys, query


def _load_data(path: Path, st, first_key: str | None) -> object:
    key = (os.path.realpath(path), st.st_mtime_ns, first_key)
    if key in _data_cache:
        return _data_cache[key]
    try:
        text = path.read_text()
    except OSError as err:
        raise CredentialsError(f"{path}: cannot read credentials file: {err.strerror}") from None
    problem = None
    data = None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        problem = (mark.line + 1, mark.column + 1) if mark is not None else ()
    # Raised outside the except block: the YAML error quotes the offending
    # line, which may be the password line, and must not become __context__.
    if problem is not None:
        where = f" at line {problem[0]}, column {problem[1]}" if problem else ""
        raise CredentialsError(f"{path}: malformed credentials record{where}")

    if isinstance(data, Mapping) and "sops" in data:
        data = _decrypt(path, first_key)
    elif st.st_mode & 0o077:
        raise CredentialsError(
            f"{path}: plaintext credentials file is accessible by group/others; "
            f"encrypt it with sops or `chmod 600 {path}`"
        )
    _data_cache[key] = data
    return data


def _merge_path(data, keys: list[str], source: str) -> dict:
    """Merge the non-mapping fields of each level along `keys`, outer to inner;
    with no keys the whole document is the record."""
    node = data
    levels = [data] if not keys else []
    walked = []
    for k in keys:
        walked.append(k)
        if not isinstance(node, Mapping) or k not in node:
            raise CredentialsError(f"{source}: no key '{'.'.join(walked)}' in the credentials document")
        node = node[k]
        levels.append(node)
    merged: dict = {}
    for level in levels:
        if not isinstance(level, Mapping):
            raise CredentialsError(f"{source}: '{'.'.join(keys)}' is not a mapping")
        merged.update({k: v for k, v in level.items() if not isinstance(v, Mapping)})
    return merged


def _materialise_ca(pem, source: str) -> str:
    if not isinstance(pem, str) or "-----BEGIN" not in pem:
        raise CredentialsError(f"{source}: key 'ca' must be a PEM certificate (-----BEGIN ...)")
    digest = hashlib.sha256(pem.encode()).hexdigest()
    if digest not in _ca_files or not os.path.exists(_ca_files[digest]):
        if not _ca_dir:
            d = tempfile.mkdtemp(dir=os.environ.get("XDG_RUNTIME_DIR") or None)  # 0700
            _ca_dir.append(d)
            atexit.register(shutil.rmtree, d, ignore_errors=True)
        file = os.path.join(_ca_dir[0], f"{digest}.pem")
        fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(pem)
        _ca_files[digest] = file
    return _ca_files[digest]


def _build(data, keys: list[str], query: str, source: str) -> PgCredentials:
    rec = _merge_path(data, keys, source)
    for alias, canonical in _ALIASES.items():
        if alias in rec:
            if canonical in rec:
                raise CredentialsError(f"{source}: keys '{alias}' and '{canonical}' are the same field; set one")
            rec[canonical] = rec.pop(alias)
    if "ca" in rec:
        # Overrides ca_path/sslrootcert: a `ca_path` in a shared record is
        # typically another consumer's container path, absent on this host.
        rec["sslrootcert"] = _materialise_ca(rec.pop("ca"), source)
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True) if query else []
    except ValueError:
        raise CredentialsError(f"{source}: malformed credentials query '?{query}'") from None
    for name, value in pairs:
        if name in _SECRET_FIELDS:
            raise CredentialsError(f"{source}: '{name}' is a secret and cannot be set in the reference query")
        if name not in _OVERRIDABLE:
            raise CredentialsError(f"{source}: unknown reference query key '{name}'")
        rec[name] = value
    if rec.get("sslrootcert") is not None and rec.get("sslmode") is None:
        rec["sslmode"] = "verify-full"
    return PgCredentials.from_mapping(rec, source=source)


def load_credentials(reference) -> PgCredentials:
    ref = str(reference)
    path_s, keys, query = _split_reference(ref)
    path = Path(path_s)
    try:
        st = os.stat(path)
    except OSError as err:
        raise CredentialsError(f"{path}: cannot read credentials file: {err.strerror}") from None
    key = (os.path.realpath(path), st.st_mtime_ns, ref)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    data = _load_data(path, st, keys[0] if keys else None)
    creds = _build(data, keys, query, str(path))
    _cache[key] = creds
    return creds


def _cache_clear() -> None:
    _cache.clear()
    _data_cache.clear()


load_credentials.cache_clear = _cache_clear  # type: ignore[attr-defined]


def resolve_url(*, credentials=None, dsn=None):
    if (credentials is None) == (dsn is None):
        raise CredentialsError("set exactly one of credentials and dsn")
    if credentials is not None:
        return load_credentials(credentials).url()
    return dsn
