import pytest

from ultralight_rag.auth import AuthenticationError, AuthorizationError, Authorizer, Principal
from ultralight_rag.config import Settings


def test_no_auth_mode_grants_every_action_as_anonymous_admin():
    authorizer = Authorizer(Settings())
    for action in ("read", "write", "delete"):
        assert authorizer.authorize({}, action) == Principal(user="anonymous", role="admin")


def test_no_auth_mode_ignores_action_value_and_still_grants_anonymous_admin():
    authorizer = Authorizer(Settings())
    assert authorizer.authorize({}, "totally-unrecognized-action") == Principal(
        user="anonymous", role="admin"
    )


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
    assert authorizer.authorize(
        {"X-User": "reader@example.test", "X-Role": "viewer"}, "read"
    ) == Principal(user="reader@example.test", role="reader")
    assert authorizer.authorize(
        {"X-User": "admin@example.test", "X-Role": "owner"}, "delete"
    ) == Principal(user="admin@example.test", role="admin")


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


def test_trusted_proxy_rejects_empty_or_ambiguous_role_configuration():
    with pytest.raises(ValueError, match="role"):
        Authorizer(Settings(auth_mode="trusted-proxy", proxy_admin_role=""))
    with pytest.raises(ValueError, match="distinct"):
        Authorizer(
            Settings(
                auth_mode="trusted-proxy",
                proxy_admin_role="same",
                proxy_reader_role="same",
            )
        )
    with pytest.raises(ValueError, match="header"):
        Authorizer(Settings(auth_mode="trusted-proxy", proxy_user_header=" "))


def test_trusted_proxy_strips_surrounding_whitespace_from_header_values():
    """Proxies may emit header values padded with whitespace.

    Both the identity and the role are stripped before use: an unstripped role
    would never match the configured value and would lock out a legitimate
    admin, and an unstripped user would carry the padding into the audit log.
    """
    authorizer = Authorizer(
        Settings(
            auth_mode="trusted-proxy",
            proxy_user_header="X-User",
            proxy_role_header="X-Role",
            proxy_admin_role="owner",
            proxy_reader_role="viewer",
        )
    )

    principal = authorizer.authorize(
        {"X-User": "  admin@example.test\t", "X-Role": " owner "}, "write"
    )

    assert principal == Principal(user="admin@example.test", role="admin")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("none", "none"),
        ("NONE", "none"),
        ("None", "none"),
        ("trusted-proxy", "trusted-proxy"),
        ("trusted_proxy", "trusted-proxy"),
        ("Trusted_Proxy", "trusted-proxy"),
        ("TRUSTED-PROXY", "trusted-proxy"),
    ],
)
def test_auth_mode_is_normalized_for_case_and_underscores(configured, expected):
    """RAG_AUTH_MODE accepts case and underscore variants.

    The env var invites snake_case, so `trusted_proxy` must resolve the same as
    the documented `trusted-proxy`; getting a ValueError at startup for that
    would be a hostile surprise. Neither half of the normalization was pinned:
    dropping `.lower()` or `.replace("_", "-")` each left all 161 tests green.
    """
    assert Authorizer(Settings(auth_mode=configured)).mode == expected
