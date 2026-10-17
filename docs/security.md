# Security, Threat Model & Privacy Controls

## Security Objective

PrivateDocs-LLM is engineered as a zero-data-egress reference architecture for confidential internal documents. All document ingestion, vector retrieval, and model inference execute strictly within the local host boundary or isolated cloud VM without telemetry or third-party API transmissions.

## Implemented Protections

1. **API Key Authentication**:
   - API keys are hashed using `SHA-256` before verification against a local SQLite key store.
   - Separate role-based permissions: `admin` (document ingestion and management) and `user` (querying).
   - Keys are never logged in plaintext.

2. **Per-Key Rate Limiting**:
   - In-memory rate limiting prevents denial-of-service and resource exhaustion.
   - Returns HTTP 429 with explicit `Retry-After` headers.

3. **Privacy-Preserving Logging**:
   - Request and document content are **never logged at INFO level**.
   - Application logs record only request UUIDs, token counts, character lengths, and latency metrics.

4. **Network Isolation**:
   - No management or UI ports (Streamlit, Grafana, vLLM) are bound to public network interfaces in production.
   - Remote access is restricted to secure SSH tunneling (`ssh -L`).

5. **Secrets Hygiene**:
   - API keys and tokens are loaded strictly from environment variables or `.env` files (which are git-ignored).

---

## Known Gaps & Out-of-Scope (Honest Limitations)

This repository is a **single-node reference architecture**, not a multi-tenant certified compliant enterprise SaaS. The following protections are explicitly out-of-scope for this phase:

- **Per-Document Access Control Lists (ACLs)**: Any authenticated user with the `user` role can retrieve information across all indexed documents in the active Qdrant collection.
- **Vector DB Encryption at Rest**: Default Qdrant storage is unencrypted at the application layer; disk-level encryption (LUKS/dm-crypt) must be provided by the host OS.
- **Audit Log Tamper-Resistance**: Standard file/stdout logging without cryptographic hash-chaining or remote append-only log forwarders.
- **Adversarial Prompt Injection**: Prompt templates use citation-grounding instructions, but do not provide formal mathematical guarantees against sophisticated jailbreaks or indirect prompt injections embedded in ingested files.
- **High Availability & Distributed Clustering**: Single container / instance deployment without multi-node leader election or database replication.
