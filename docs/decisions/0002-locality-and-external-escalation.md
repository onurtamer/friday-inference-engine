# ADR 0002: Locality, credentials, and external escalation

- Status: accepted; implementation deferred to v2 (see ADR 0003)
- Date: 2026-10-27

## Context

FRIDAY must keep user and project data local by default. External AI may be
used only for selected difficult work, when data is sanitized and the user
explicitly approves the particular request. Development-time use of AI
assistants is separate from FRIDAY's runtime architecture.

## Decision

- The default runtime has no external API provider enabled and no network
  egress path.
- Any future external gateway is a separate, opt-in component.
- Each request requires user approval that identifies destination/service,
  purpose, and the sanitized fields to be transmitted.
- There is no automatic escalation to an external model (for example after
  repeated local failures).
- Credentials are handled by a local broker or OS-backed vault and injected
  only at request time. They do not enter model context or routine logs.
- If approval is denied, local execution or a deferred result is used.
- External model output is untrusted input and cannot modify the repository
  directly.

## Consequences

- Local tasks must have a useful offline path.
- External calls are auditable and can be disabled independently.
- This ADR does not authorize any API call or credential creation.
