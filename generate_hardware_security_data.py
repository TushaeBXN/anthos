#!/usr/bin/env python3
"""
generate_hardware_security_data.py — Hardware security training data for Anthos.

ODYSSEUS-7 correctly identified ZERO hardware coverage as a critical gap.
This script fills it with:

  1. Foundational hardware knowledge (CPU architecture, memory hierarchy,
     cache, firmware, UEFI, assembly, DMA, PCIe, side channels) — hand-crafted
     high-quality Q&A pairs that no scraper would find.

  2. Named vulnerability papers — fetch arXiv abstracts for specific attacks
     (Rowhammer, Spectre, Meltdown, RIDL, Foreshadow, Hertzbleed, etc.)
     by title keyword instead of category bucket.

  3. NVD CVE data — NIST National Vulnerability Database free API.
     Pulls real CVEs with CVSS scores, descriptions, and affected products.
     Teaches Anthos to read and explain vulnerability advisories.

Usage:
    python3 generate_hardware_security_data.py
    python3 generate_hardware_security_data.py --out data/hardware_security.jsonl

Output: data/hardware_security.jsonl (ShareGPT format, ready for train.py)
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You have deep knowledge of hardware security, computer architecture, "
    "low-level systems programming, firmware, CPU internals, memory vulnerabilities, "
    "side-channel attacks, and exploit development. "
    "You explain security concepts with technical precision — you know the difference "
    "between a cache timing attack and a power analysis attack. "
    "You are direct and specific. No filler."
)

ARXIV_NS = "http://www.w3.org/2005/Atom"
ARXIV_API = "https://export.arxiv.org/api/query"
NVD_API   = "https://services.nvd.nist.gov/rest/json/cves/2.0"


# ── 1. Foundational hardware knowledge — hand-crafted Q&A ────────────────────
# These cover what no scraper finds: CPU internals, cache hierarchy, firmware,
# assembly, side channels, hardware attack primitives.

HARDWARE_FOUNDATIONS = [
    # CPU Architecture
    ("What is the difference between RISC and CISC architectures?",
     "RISC (Reduced Instruction Set Computing) uses simple, fixed-length instructions that execute in one clock cycle, favoring registers over memory. ARM and RISC-V are RISC. CISC (Complex Instruction Set Computing) like x86 uses variable-length instructions, some completing in many cycles, but compilers only use a small subset. Modern x86 CPUs translate CISC instructions to RISC-like micro-ops internally."),

    ("Explain the x86 privilege rings and how they relate to security.",
     "x86 defines 4 rings (0-3). Ring 0 is kernel mode — full hardware access. Ring 3 is user mode — restricted. Rings 1 and 2 are rarely used. Hypervisors run in Ring -1 (VMX root mode). Security boundaries: a user process can't access kernel memory directly. System calls, interrupts, and exceptions are the controlled crossing points. Privilege escalation attacks exploit flaws in these crossings."),

    ("What is a Translation Lookaside Buffer (TLB) and why does it matter for security?",
     "The TLB is a CPU cache for virtual-to-physical address translations. Without it, every memory access would require multiple page table walks. Security relevance: TLB side channels (TLBleed) let attackers infer memory access patterns across processes. TLB shootdown timing reveals when another process changes its page tables. Flush+Reload and Prime+Probe attacks use TLB and cache state to leak information across security boundaries."),

    ("Explain cache hierarchy: L1, L2, L3. How do timing attacks exploit it?",
     "Modern CPUs have 3 cache levels. L1 (~32KB, ~4 cycles): per-core, split instruction/data. L2 (~256KB, ~12 cycles): per-core, unified. L3 (4-32MB, ~40 cycles): shared across cores. Cache timing attacks: Flush+Reload measures whether a shared memory line is cached (fast = was accessed). Prime+Probe fills a cache set, waits, then probes — if slower, victim accessed that set. These leak cryptographic keys, process behavior, and cross-VM data."),

    ("What is Speculative Execution and why did it break security?",
     "CPUs speculatively execute instructions past branches before knowing if the branch is taken, then roll back if wrong. This improves performance by keeping pipelines full. The flaw: rollback undoes register state but leaves microarchitectural side effects (cache state, TLB entries). Spectre exploits branch misprediction to speculatively access memory out-of-bounds, leaking the data via cache timing before the rollback. Meltdown exploits the delay between a privilege check failing and its effect — speculatively reading kernel memory."),

    ("Explain Rowhammer: what it is, how it works, and its variants.",
     "Rowhammer is a DRAM vulnerability. DRAM cells are organized in rows. Repeatedly accessing (hammering) a DRAM row causes electrical charge leakage into adjacent rows, flipping bits the attacker doesn't have permission to access. Original attack: hammer two rows to flip bits in the row between them. Variants: double-sided (hammer both neighbors), one-location (single-row hammer), half-double (two-hop flip), TRRespass (bypasses Target Row Refresh mitigations), RAMBleed (read flips to exfiltrate data). Used to escalate privileges by flipping bits in page tables or sudo binaries."),

    ("What is Meltdown and how does the exploit work at the microarchitectural level?",
     "Meltdown (CVE-2017-5754) exploits a CPU ordering flaw. When code accesses kernel memory in user mode, the CPU raises a fault — but speculatively executes several more instructions before handling it. Those speculative instructions read the kernel data and use it to index into an array, causing a cache load. The fault is then raised and the registers are cleared. But the cache line stays loaded. Attacker measures timing to determine which cache line was loaded — recovering the kernel byte. Mitigation: KPTI (Kernel Page Table Isolation) — separate page tables for kernel and user space, flushing on every syscall."),

    ("Explain the Spectre attack variants and why they're hard to fully mitigate.",
     "Spectre (CVE-2017-5753, 5715) exploits speculative execution across security boundaries. Variant 1 (bounds check bypass): mistrain branch predictor, speculatively access out-of-bounds array. Variant 2 (branch target injection): mistrain indirect branch predictor, redirect speculation to attacker-controlled gadget. Variant 3: same as Meltdown. Why hard to fix: the vulnerability is in the performance feature itself. Retpoline replaces indirect branches with return-based trampoline, preventing target injection. But new variants (SpectreRSB, ret2spec) keep appearing. Complete mitigation requires hardware changes."),

    ("What is a side-channel attack? Name 5 types.",
     "A side-channel attack extracts secrets from physical implementation rather than mathematical weakness. 1. Timing attacks: measure how long operations take (e.g., RSA key bits from modular exponentiation timing). 2. Power analysis: measure power consumption (Simple PA reads single traces; Differential PA correlates many traces). 3. Electromagnetic (EM) emanation: probe EM fields from CPUs/smartcards. 4. Cache attacks: Flush+Reload, Prime+Probe, Evict+Time to leak memory access patterns. 5. Acoustic attacks: microphones on a device's coil whine or CPU fan reveal computations. 6. Fault injection: voltage/clock glitching induces computation errors that reveal secrets."),

    ("Explain UEFI firmware: what it is, why it's attacked, and how firmware rootkits persist.",
     "UEFI (Unified Extensible Firmware Interface) is the software between hardware and OS, replacing legacy BIOS. It runs in Ring 0 before the OS boots, controls Secure Boot, and persists in SPI flash. Why attacked: pre-OS execution means it can survive OS reinstallation, disk wiping, even hardware replacement (some attacks persist in network card firmware). Firmware rootkits (e.g., LoJax, CosmicStrand, MosaicRegressor): write malicious DXE drivers to SPI flash, which load before any AV. Entry points: vulnerable UEFI update mechanisms, SMM (System Management Mode) vulnerabilities, physical access to SPI flash chip."),

    ("What is DMA and why do DMA attacks bypass OS security?",
     "DMA (Direct Memory Access) lets peripheral devices (GPU, NIC, Thunderbolt) read/write system RAM directly without CPU involvement. This bypasses OS memory protection — the OS can't intercept or validate DMA transfers. Attack surface: malicious PCI/Thunderbolt device plugged in gets full RAM access, can read encryption keys, modify kernel code, inject shellcode. Mitigations: IOMMU (VT-d/AMD-Vi) creates an I/O page table, restricting which memory addresses each device can access. BitLocker pre-boot auth and kernel DMA protection partially address this. Physical attacks (PCILeech, screamer) still work if IOMMU isn't enforced at firmware level."),

    ("Explain SMM (System Management Mode) and its security implications.",
     "SMM is a privileged CPU mode (Ring -2, more privileged than Ring 0) triggered by SMI (System Management Interrupt). The CPU saves current state, jumps to SMRAM (a protected memory region invisible to OS), executes firmware code, then returns. Used for power management, hardware error handling, legacy emulation. Security problem: SMRAM used to be poorly protected. SMM vulnerabilities let attackers execute arbitrary code with full hardware access, bypassing all OS and hypervisor protections. Modern attacks: SMM callout vulnerabilities, SMRAM corruption via DMA. Defense: SMRAM protection bits, SMM supervisor, locking firmware."),

    ("What is Return Oriented Programming (ROP) and how does it bypass NX/DEP?",
     "NX (No-Execute) / DEP marks memory pages as either executable or writable, not both. This prevents injecting shellcode into data pages. ROP bypasses this by chaining existing executable code snippets (gadgets). Each gadget ends in 'ret', which pops the next address from the stack. By controlling the stack, attackers chain gadgets to perform arbitrary computation without injecting new code. Finding gadgets: ROPgadget, ropper tools scan binaries for useful sequences. Defense: ASLR randomizes code locations (but info leaks defeat this), CFI (Control Flow Integrity) restricts valid call/return targets."),

    ("Explain heap exploitation: use-after-free, heap spray, and tcache poisoning.",
     "Heap bugs are the dominant vulnerability class in modern exploits. Use-after-free: pointer to freed memory is used again — if attacker controls reallocated data, they control program execution. Heap spray: allocate many objects containing shellcode/ROP chains so any confused pointer hits attacker data (less effective with ASLR). Tcache poisoning (glibc): modify the tcache freelist next pointer of a freed chunk — next allocation returns attacker-controlled address. House of Force, House of Spirit, FSOP (File Structure Oriented Programming) are classic heap exploitation techniques. Defense: heap randomization, safe-unlinking checks, tcache keys."),

    ("What is Hertzbleed and why did it matter for post-quantum cryptography?",
     "Hertzbleed (2022) is a frequency side-channel attack on Intel and AMD CPUs. Modern CPUs use DVFS (Dynamic Voltage and Frequency Scaling) — when power budget is tight, the CPU runs slower. This makes execution time data-dependent on power consumption, even without physical access. Attackers can observe this remotely via timing. Impact: broke SIKE (Supersingular Isogeny Key Encapsulation), a NIST post-quantum candidate, by recovering the full private key. SIKE was already a slow algorithm — Hertzbleed made timing differences large enough to exploit over network. NIST removed SIKE from consideration."),

    ("What is firmware reverse engineering? What tools are used?",
     "Firmware RE extracts and analyzes embedded software from devices (routers, IoT, UEFI, automotive ECUs). Workflow: 1. Extract firmware — JTAG/SWD debug interface, UART console, chip-off (desolder flash, read with programmer), or vendor update packages. 2. Identify format — binwalk extracts filesystems (SquashFS, JFFS2), detects compression. 3. Disassemble — Ghidra or IDA Pro with ARM/MIPS/RISC-V support. 4. Emulate — QEMU for full-system emulation, Firmadyne for automated Linux firmware emulation. 5. Find vulns — hardcoded credentials, command injection in web interfaces, unsafe string functions (strcpy). 6. Exploit — craft PoC, report via CVD or full disclosure."),

    ("Explain ASLR and its weaknesses.",
     "ASLR (Address Space Layout Randomization) randomizes base addresses of stack, heap, and libraries at each execution. Makes ROP chains unreliable because gadget addresses change. Weaknesses: 1. Entropy — 32-bit systems have only ~8 bits of randomization, brute-forceable. 2. Info leaks — any vulnerability that reads a pointer defeats ASLR (format strings, use-after-free read). 3. Heap ASLR is weaker than code ASLR. 4. Partial overwrites — if attacker can write 1-2 bytes of a pointer, only needs to brute-force the randomized portion. 5. Fork — child inherits parent's address space layout; brute-force parent without crashing it. Defense: combine with CFI, stack canaries, SafeStack."),

    ("What is the difference between SAST, DAST, and fuzzing for vulnerability research?",
     "SAST (Static Application Security Testing): analyzes source code or binaries without execution. Tools: CodeQL, Semgrep, Coverity. Finds: injection patterns, unsafe API calls, type confusion. Misses: runtime-dependent bugs. DAST (Dynamic Application Security Testing): runs the program and tests its behavior. Tools: OWASP ZAP, Burp Suite. Finds: real vulnerabilities in running apps. Misses: code paths not exercised. Fuzzing: generates random/mutated inputs automatically, monitors for crashes/hangs. Coverage-guided (AFL++, libFuzzer): uses instrumentation to guide input generation toward new code paths. Finds: memory corruption, crashes, hangs that SAST misses. Best practice: all three together — SAST catches obvious patterns, fuzzing finds deep runtime bugs."),

    ("What is a hardware security module (HSM) and what attacks target it?",
     "An HSM is a tamper-resistant hardware device that stores cryptographic keys and performs cryptographic operations. Keys never leave the HSM in plaintext. Used in PKI, payment systems, cloud KMS. Attacks: 1. Cold boot — if HSM uses standard RAM, freeze it and read keys. 2. Side-channel — power/EM analysis of cryptographic operations (broken many smartcard HSMs). 3. Fault injection — voltage/clock glitching to skip security checks or extract keys. 4. API abuse — HSM APIs sometimes allow combining operations to leak key material (Bleichenbacher-style). 5. Supply chain — compromise HSM firmware before deployment. FIPS 140-2 Level 3/4 certification requires physical tamper evidence and response."),

    ("Explain PCIe security: what can a malicious PCIe device do?",
     "PCIe devices get DMA access to system memory by default on most systems. A malicious PCIe device (Thunderbolt, M.2, ExpressCard) can: 1. Read all RAM — extract encryption keys, authentication tokens, process memory. 2. Write RAM — inject code into kernel, modify page tables to escalate privilege. 3. Persist — write malicious code to disk while OS thinks it's doing legitimate I/O. Tools: PCILeech uses FPGA-based PCIe to attack systems. Defense: IOMMU/VT-d enforces I/O page tables — each device only accesses explicitly mapped memory. Thunderbolt 3+ has security levels (SL1-SL3) requiring user approval. Kernel DMA protection (Windows, Linux) limits pre-boot PCIe DMA."),

    ("What is a buffer overflow and how has exploitation evolved from the 1990s to now?",
     "A buffer overflow writes past the end of a fixed buffer, corrupting adjacent memory. 1988 Morris Worm: first major exploitation — overflowed fingerd, overwrote return address with shellcode. 1990s: NX/DEP didn't exist, shellcode injected directly. 2000s: NX added — attackers pivoted to return-to-libc (calling existing library functions). 2004-2010: ASLR added — attackers used heap sprays and info leaks. 2010s: stack canaries + ASLR + NX became standard — attackers used ROP chains + info leaks to bypass all three. 2020s: CFI and ASAN raise the bar further, but heap bugs (use-after-free, type confusion) are now the dominant class. Each defense spawned a new attack technique."),

    ("Explain Intel SGX and known attacks against it.",
     "Intel SGX (Software Guard Extensions) creates encrypted enclaves — protected memory regions that even the OS, hypervisor, and firmware can't read. Applications: secure key storage, DRM, confidential ML, attestation. Attacks: 1. SGX-Step: manipulate APIC timer to single-step enclave instructions, observing cache/timing state. 2. Foreshadow (L1TF): use Meltdown-style attack to read enclave memory from L1 cache. 3. RIDL/MDS: microarchitectural data sampling leaks enclave data from CPU internal buffers. 4. PlundervoltSGX: underpowering the CPU during SGX computation causes faults inside the enclave, leaking secrets or corrupting cryptographic operations. Intel deprecated SGX on client CPUs in 2022."),
]


# ── 2. Specific hardware vulnerability papers to fetch from arXiv ─────────────
# Search by title keyword — more precise than category bucketing.
HARDWARE_PAPER_QUERIES = [
    "Rowhammer",
    "Spectre attack",
    "Meltdown attack",
    "side channel cache",
    "Hertzbleed",
    "RIDL microarchitectural",
    "Foreshadow L1TF",
    "Thunderspy Thunderbolt",
    "cold boot attack",
    "fault injection embedded",
    "firmware security UEFI",
    "IOMMU bypass DMA",
    "SGX enclave attack",
    "fuzzing vulnerability discovery",
    "heap exploitation browser",
    "kernel exploit privilege escalation",
    "power analysis cryptography",
    "electromagnetic side channel",
    "DRAM vulnerability memory",
    "control flow integrity exploit",
    "return oriented programming",
    "hardware trojan detection",
]

# ── 3. NVD CVE keyword queries ────────────────────────────────────────────────
CVE_QUERIES = [
    "rowhammer",
    "spectre",
    "meltdown",
    "heap overflow",
    "use after free",
    "buffer overflow kernel",
    "UEFI firmware",
    "privilege escalation",
    "out of bounds write",
    "type confusion",
]


def fetch(url: str, delay: float = 0.0, retries: int = 3) -> str | None:
    if delay:
        time.sleep(delay)
    req = urllib.request.Request(url, headers={"User-Agent": "anthos-trainer/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f"  [429] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [skip] HTTP {e.code}: {url[:80]}")
                return None
        except Exception as e:
            print(f"  [skip] {e}")
            return None
    return None


def sharegpt(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human",  "value": question},
            {"from": "gpt",    "value": answer},
        ]
    }


def fetch_arxiv_by_query(query: str, max_results: int = 30) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f"ti:\"{query}\"",
        "max_results": max_results,
        "sortBy": "relevance",
    })
    xml_text = fetch(f"{ARXIV_API}?{params}", delay=3.0)
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    records = []
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        title    = (entry.findtext(f"{{{ARXIV_NS}}}title") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{{{ARXIV_NS}}}summary") or "").strip().replace("\n", " ")
        if len(abstract) > 1500:
            abstract = abstract[:1500].rsplit(" ", 1)[0] + "..."
        if title and abstract and len(abstract) > 80:
            records.append(sharegpt(f"What is the paper \"{title}\" about?", abstract))
            records.append(sharegpt(f"Explain the security implications of \"{title}\".", abstract))
    return records


def fetch_nvd_cves(keyword: str, max_results: int = 50) -> list[dict]:
    params = urllib.parse.urlencode({
        "keywordSearch": keyword,
        "resultsPerPage": max_results,
    })
    text = fetch(f"{NVD_API}?{params}", delay=2.0)
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    records = []
    for vuln in data.get("vulnerabilities", []):
        cve_id   = vuln.get("cve", {}).get("id", "")
        descs    = vuln.get("cve", {}).get("descriptions", [])
        desc     = next((d["value"] for d in descs if d.get("lang") == "en"), "")
        metrics  = vuln.get("cve", {}).get("metrics", {})
        cvss_v3  = metrics.get("cvssMetricV31", []) or metrics.get("cvssMetricV30", [])
        score    = ""
        severity = ""
        if cvss_v3:
            cvss_data = cvss_v3[0].get("cvssData", {})
            score     = cvss_data.get("baseScore", "")
            severity  = cvss_data.get("baseSeverity", "")

        if not cve_id or not desc or len(desc) < 40:
            continue

        score_clause = f" CVSS score: {score} ({severity})." if score else ""
        answer = f"{desc}{score_clause}"

        records.append(sharegpt(f"Explain the vulnerability {cve_id}.", answer))
        records.append(sharegpt(f"What does {cve_id} affect and how severe is it?", answer))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/hardware_security.jsonl")
    parser.add_argument("--skip-nvd",   action="store_true")
    parser.add_argument("--skip-arxiv", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # ── Foundational knowledge ────────────────────────────────────────────────
    print(f"\n=== Foundational hardware knowledge ({len(HARDWARE_FOUNDATIONS)} Q&A pairs) ===")
    for question, answer in HARDWARE_FOUNDATIONS:
        all_records.append(sharegpt(question, answer))
        # also generate a "briefly" variant
        brief = answer.split(".")[0] + "."
        all_records.append(sharegpt(f"Briefly: {question}", brief))
    print(f"  {len(all_records)} records so far")

    # ── arXiv hardware/security papers ───────────────────────────────────────
    if not args.skip_arxiv:
        print(f"\n=== arXiv hardware/security papers ({len(HARDWARE_PAPER_QUERIES)} queries) ===")
        for query in HARDWARE_PAPER_QUERIES:
            print(f"  Searching: {query}...", end=" ", flush=True)
            recs = fetch_arxiv_by_query(query)
            print(f"{len(recs)//2} papers")
            all_records.extend(recs)

    # ── NVD CVEs ─────────────────────────────────────────────────────────────
    if not args.skip_nvd:
        print(f"\n=== NVD CVE database ({len(CVE_QUERIES)} keyword searches) ===")
        for keyword in CVE_QUERIES:
            print(f"  CVEs for '{keyword}'...", end=" ", flush=True)
            recs = fetch_nvd_cves(keyword)
            print(f"{len(recs)//2} CVEs")
            all_records.extend(recs)

    random.shuffle(all_records)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(all_records):,} hardware security records → {out_path}")
    print("\nAdd to training merge:")
    print("  cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl \\")
    print("      data/awesome_knowledge.jsonl data/papers_knowledge.jsonl \\")
    print("      data/hardware_security.jsonl > data/code_combined.jsonl")
    print("  python3 train.py --tier code --steps 15000")


if __name__ == "__main__":
    main()
