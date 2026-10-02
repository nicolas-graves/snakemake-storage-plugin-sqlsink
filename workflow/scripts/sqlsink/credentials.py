"""PostgreSQL credentials loaded from a structured, SOPS-encrypted record.

A record is a YAML/JSON mapping with separate fields (host, dbname, user,
password, ...) instead of a raw DSN, so the password never travels inside a
URL string that gets logged or echoed. An encrypted record (top-level `sops`
key) is decrypted by shelling out to `sops`; a plaintext record is only
accepted when it is private to its owner, like ~/.pgpass.

Error messages name the source and the offending KEYS, never any value.
This is a leaf module: stdlib, yaml and sqlalchemy only.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from sqlalchemy.engine import URL

# Optional libpq connection options; passed to the driver as URL query keys.
_QUERY_KEYS = ("sslmode", "sslrootcert", "sslcert", "sslkey", "options")


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
# each file once. Keyed on mtime so an edited/rotated record is picked up.
_cache: dict[tuple[str, int], PgCredentials] = {}


def _decrypt(path: Path) -> object:
    try:
        proc = subprocess.run(
            ["sops", "--decrypt", "--output-type", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
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
        return json.loads(proc.stdout)
    except ValueError:
        raise CredentialsError(f"{path}: sops produced output that is not valid JSON") from None


def load_credentials(path) -> PgCredentials:
    path = Path(path)
    try:
        st = os.stat(path)
        text = path.read_text()
    except OSError as err:
        raise CredentialsError(f"{path}: cannot read credentials file: {err.strerror}") from None
    key = (os.path.realpath(path), st.st_mtime_ns)
    cached = _cache.get(key)
    if cached is not None:
        return cached

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
        data = _decrypt(path)
    elif st.st_mode & 0o077:
        raise CredentialsError(
            f"{path}: plaintext credentials file is accessible by group/others; "
            f"encrypt it with sops or `chmod 600 {path}`"
        )
    creds = PgCredentials.from_mapping(data, source=str(path))
    _cache[key] = creds
    return creds


load_credentials.cache_clear = _cache.clear  # type: ignore[attr-defined]


def resolve_url(*, credentials=None, dsn=None):
    if (credentials is None) == (dsn is None):
        raise CredentialsError("set exactly one of credentials and dsn")
    if credentials is not None:
        return load_credentials(credentials).url()
    return dsn
