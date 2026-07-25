import pytest

from utralight_rag.auth import AuthenticationError, AuthorizationError, Authorizer
from utralight_rag.config import Settings


def test_no_auth_mode_grants_every_action():
    authorizer = Authorizer(Settings())
    for action in ("read", "write", "delete"):
        assert authorizer.authorize({}, action).role == "admin"


def test_trusted_proxy_requires_identity_and_maps_roles():
    authorizer = Authorizer(
        Settings(
            auth_mode="trusted-proxy",
            proxy_user_header="X-User",
            proxy_role_header="X-Role",
            proxy_admin_role="owner",
            proxy_reader_role="viewer",
        )
    )
    with pytest.raises(AuthenticationError):
        authorizer.authorize({}, "read")
    with pytest.raises(AuthorizationError):
        authorizer.authorize({"X-User": "reader@example.test", "X-Role": "viewer"}, "write")
    assert (
        authorizer.authorize({"X-User": "reader@example.test", "X-Role": "viewer"}, "read").role
        == "reader"
    )
    assert (
        authorizer.authorize({"X-User": "admin@example.test", "X-Role": "owner"}, "delete").role
        == "admin"
    )


def test_trusted_proxy_rejects_unknown_roles():
    authorizer = Authorizer(Settings(auth_mode="trusted-proxy"))
    with pytest.raises(AuthorizationError):
        authorizer.authorize(
            {
                "Cf-Access-Authenticated-User-Email": "user@example.test",
                "X-Auth-Request-Role": "guest",
            },
            "read",
        )


def test_authorizer_rejects_unknown_mode():
    with pytest.raises(ValueError, match="RAG_AUTH_MODE"):
        Authorizer(Settings(auth_mode="unsupported"))
