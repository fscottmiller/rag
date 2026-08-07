"""Authorization for open and trusted-proxy runtime modes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """The request does not contain a trusted proxy identity."""


class AuthorizationError(Exception):
    """The authenticated role cannot perform the requested action."""


@dataclass(frozen=True)
class Principal:
    user: str
    role: str


class Authorizer:
    """Map trusted proxy headers to the two supported application roles."""

    def __init__(self, settings: Settings) -> None:
        mode = settings.auth_mode.lower().replace("_", "-")
        if mode not in {"none", "trusted-proxy"}:
            raise ValueError("RAG_AUTH_MODE must be 'none' or 'trusted-proxy'")
        self.mode = mode
        self.user_header = settings.proxy_user_header.strip()
        self.role_header = settings.proxy_role_header.strip()
        self.admin_role = settings.proxy_admin_role.strip()
        self.reader_role = settings.proxy_reader_role.strip()
        if self.mode == "trusted-proxy":
            if not self.user_header or not self.role_header:
                raise ValueError("Trusted proxy headers must not be empty")
            if not self.admin_role or not self.reader_role:
                raise ValueError("Trusted proxy roles must not be empty")
            if self.admin_role == self.reader_role:
                raise ValueError("Trusted proxy admin and reader roles must be distinct")

    def authorize(self, headers: Mapping[str, str], action: str) -> Principal:
        if self.mode == "none":
            return Principal(user="anonymous", role="admin")

        user = self._header(headers, self.user_header)
        role = self._header(headers, self.role_header)
        if not user:
            logger.warning(
                "Authorization denied: no trusted proxy identity present, action=%s", action
            )
            raise AuthenticationError("Trusted proxy identity is required")
        if role == self.admin_role:
            return Principal(user=user, role="admin")
        if role == self.reader_role and action == "read":
            return Principal(user=user, role="reader")
        # user and role come straight from HTTP headers (see _header(), which only
        # strips whitespace) with no trust boundary in between, so an attacker fully
        # controls their content. %r escapes an embedded newline instead of letting it
        # forge a second, well-formed log line in the authorization audit trail --
        # action is an internal literal ("read"/"write"), not attacker data, so %s is
        # fine for it.
        logger.warning("Authorization denied: user=%r role=%r action=%s", user, role, action)
        raise AuthorizationError("This role is not allowed to perform this action")

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value.strip()
        return ""
