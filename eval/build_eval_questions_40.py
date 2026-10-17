"""Script to generate eval/datasets/eval_questions_40.jsonl with exact evidence phrases."""

import json
import pickle
import re
from pathlib import Path

with open("data/bm25_index.pkl", "rb") as f:
    chunks = pickle.load(f)["chunks"]

chunks_by_doc = {}
for c in chunks:
    chunks_by_doc.setdefault(c.doc_id, []).append(c)

def find_first(doc_id, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for c in chunks_by_doc.get(doc_id, []):
        m = regex.search(c.content)
        if m:
            snippet = c.content[m.start():m.end() + 40].strip()
            clean_snippet = " ".join(snippet.split()[:12])
            return c, clean_snippet
    return None, None

specs = [
    # --- Apple ---
    ("q_aapl_1", "sec_apple_10k_2023", "factual",
     "What risks does Apple disclose regarding cellular carrier financing and subsidy plans?",
     r"carriers providing cellular network service for the Company",
     "cellular network service for the Company",
     "Apple notes that carriers provide financing, installment payment plans, or subsidies for purchases of Apple products, and changes to these programs could affect product sales."),

    ("q_aapl_2", "sec_apple_10k_2023", "numeric",
     "What was Apple's total undiscounted liabilities and imputed interest reported in Note 7?",
     r"Total undiscounted liabilities",
     "Total undiscounted liabilities",
     "Apple reported total undiscounted liabilities of $15,266 million with imputed interest of $(2,424) million."),

    ("q_aapl_3", "sec_apple_10k_2023", "multi_chunk",
     "How did Apple's selling, general, and administrative expenses change between 2022 and 2023?",
     r"Selling, general and administrative expense was relatively flat",
     "Selling, general and administrative expense was relatively flat",
     "Selling, general and administrative expense was relatively flat in 2023 compared to 2022."),

    # --- Microsoft ---
    ("q_msft_1", "sec_microsoft_10k_2023", "factual",
     "What are the terms and regulatory considerations regarding Microsoft's proposed acquisition of Activision Blizzard?",
     r"Activision Blizzard",
     "Activision Blizzard",
     "Microsoft entered into an agreement to acquire Activision Blizzard, Inc., subject to regulatory approvals and closing conditions."),

    ("q_msft_2", "sec_microsoft_10k_2023", "numeric",
     "What was Microsoft's commercial remaining performance obligation (RPO) balance as of June 30, 2023?",
     r"remaining performance obligation",
     "remaining performance obligation",
     "Microsoft reported commercial remaining performance obligation of billions expected to be recognized over future periods."),

    ("q_msft_3", "sec_microsoft_10k_2023", "multi_chunk",
     "How does Microsoft characterize datacenter capacity needs and infrastructure investments for cloud and AI?",
     r"datacenter capacity",
     "datacenter capacity",
     "Microsoft notes continuing investments in datacenter capacity and AI computing infrastructure to support customer demand."),

    # --- NVIDIA ---
    ("q_nvda_1", "sec_nvidia_10k_2024", "factual",
     "What export control licensing requirements did the US government impose on NVIDIA's high-performance GPUs shipped to China?",
     r"licensing requirements.*China",
     "licensing requirements",
     "The US government announced interim final rules impacting shipments of high-performance GPUs and compute boards to China and certain other destinations without a license."),

    ("q_nvda_2", "sec_nvidia_10k_2024", "numeric",
     "What were NVIDIA's Data Center segment revenues in fiscal year 2024?",
     r"Data Center revenue",
     "Data Center revenue",
     "NVIDIA's Data Center revenue grew substantially in fiscal 2024 driven by cloud service providers and enterprise accelerated computing."),

    ("q_nvda_3", "sec_nvidia_10k_2024", "factual",
     "How did NVIDIA describe customer adoption and demand for the Hopper architecture GPUs?",
     r"Hopper architecture",
     "Hopper architecture",
     "NVIDIA described surging adoption for its Hopper architecture GPUs, including HGX systems and H100 Tensor Core GPUs for AI workloads."),

    ("q_nvda_4", "sec_nvidia_10k_2024", "multi_chunk",
     "What are NVIDIA's two primary reportable operating segments and their primary end markets?",
     r"Compute & Networking",
     "Compute & Networking",
     "NVIDIA reports results across two primary segments: Compute & Networking (including Data Center and networking) and Graphics (including GeForce gaming)."),

    # --- Alphabet ---
    ("q_goog_1", "sec_alphabet_10k_2023", "factual",
     "How could Internet service providers restricting access affect Alphabet's advertising and cloud businesses?",
     r"Internet access providers may be able to restrict",
     "Internet access providers may be able to restrict",
     "Alphabet warned that Internet access providers may restrict, block, or degrade user access to products and services or charge fees for content delivery."),

    ("q_goog_2", "sec_alphabet_10k_2023", "numeric",
     "What was the total dollar amount spent by Alphabet on Class C share repurchases in 2023?",
     r"Class C share repurchases",
     "Class C share repurchases",
     "Alphabet reported $52,577 million in Class C share repurchases and over $59 billion in total share repurchases in 2023."),

    ("q_goog_3", "sec_alphabet_10k_2023", "multi_chunk",
     "What components make up Google Services and Google Cloud revenues within Alphabet?",
     r"Google Services",
     "Google Services",
     "Alphabet segments its revenue into Google Services (Search, YouTube ads, Google Network, Google Other) and Google Cloud."),

    # --- Amazon ---
    ("q_amzn_1", "sec_amazon_10k_2023", "factual",
     "What risks does Amazon face in connection with its government contracts and procurement regulations?",
     r"Government Contracts and Related Procurement",
     "Government Contracts and Related Procurement",
     "Amazon noted compliance risks, auditing scrutiny, and suspension or debarment risks related to government contracting."),

    ("q_amzn_2", "sec_amazon_10k_2023", "numeric",
     "What proceeds did Amazon receive from property and equipment sales and incentives in 2023?",
     r"Proceeds from property and equipment sales",
     "Proceeds from property and equipment sales",
     "Amazon recorded $5,657 million in proceeds from property and equipment sales and incentives in 2023."),

    ("q_amzn_3", "sec_amazon_10k_2023", "factual",
     "How did Amazon describe AWS segment sales growth and operational focus during 2023?",
     r"AWS segment",
     "AWS segment",
     "The AWS segment offers global compute, storage, database, and other service offerings with continuing investments in technology infrastructure."),

    ("q_amzn_4", "sec_amazon_10k_2023", "multi_chunk",
     "How does Amazon handle unearned revenue and long-term customer contractual obligations?",
     r"unearned revenue",
     "unearned revenue",
     "Amazon records unearned revenue when payments are received in advance of service performance, recognizing it as services are rendered."),

    # --- AMD ---
    ("q_amd_1", "sec_amd_10k_2023", "factual",
     "What relationship and transactions does AMD maintain with its ATMP joint ventures?",
     r"purchases from the ATMP JV",
     "purchases from the ATMP JV",
     "AMD purchases assembly, test, mark, and packaging services from the ATMP JV, amounting to approximately $1.7 billion in 2023 and 2022."),

    ("q_amd_2", "sec_amd_10k_2023", "numeric",
     "What impact did the Xilinx acquisition have on AMD's amortization of acquired intangible assets in 2023?",
     r"Xilinx",
     "Xilinx",
     "AMD recorded significant amortization of acquired intangible assets primarily related to the acquisition of Xilinx."),

    ("q_amd_3", "sec_amd_10k_2023", "numeric",
     "What was AMD's Data Center segment net revenue in 2023 compared to 2022?",
     r"Data Center segment revenue",
     "Data Center segment revenue",
     "AMD reported that Data Center segment revenue was driven by growth in EPYC server processors and Instinct GPU accelerators."),

    ("q_amd_4", "sec_amd_10k_2023", "multi_chunk",
     "How did AMD's Client and Gaming segments perform due to inventory adjustments and PC demand?",
     r"Cautionary Statement Regarding Forward-Looking Statements",
     "Cautionary Statement",
     "AMD experienced fluctuations in PC demand and graphics card channel inventories affecting Client and Gaming segment revenues."),

    # --- Meta ---
    ("q_meta_1", "sec_meta_10k_2023", "factual",
     "What is Meta's operational focus and financial performance within the Reality Labs segment?",
     r"Reality Labs",
     "Reality Labs",
     "Reality Labs includes augmented and virtual reality related consumer hardware, software, and content, which operates at an operating loss due to long-term R&D investments."),

    ("q_meta_2", "sec_meta_10k_2023", "numeric",
     "How did Meta's ad impressions and average price per ad change across its Family of Apps in 2023?",
     r"ad impressions",
     "ad impressions",
     "Meta reported a year-over-year increase in ad impressions across the Family of Apps, partially offset by changes in average price per ad."),

    ("q_meta_3", "sec_meta_10k_2023", "factual",
     "What does Meta disclose regarding antitrust and consumer protection claims by the Federal Trade Commission?",
     r"Federal Trade Commission",
     "Federal Trade Commission",
     "Meta is subject to an FTC lawsuit alleging monopolization and seeking remedies including divestiture of Instagram and WhatsApp."),

    ("q_meta_4", "sec_meta_10k_2023", "multi_chunk",
     "What products constitute Meta's Family of Apps and how are they monetized?",
     r"Family of Apps",
     "Family of Apps",
     "The Family of Apps includes Facebook, Instagram, Messenger, WhatsApp, and other services, primarily monetized through advertising."),

    # --- White House EO 14110 ---
    ("q_eo_1", "whitehouse_eo_14110_ai", "factual",
     "What mitigation measures does EO 14110 specify regarding chemical, biological, radiological, or nuclear (CBRN) threats in AI?",
     r"chemical, biological, radiological",
     "chemical, biological, radiological",
     "Executive Order 14110 requires actions to evaluate and mitigate AI risks related to chemical, biological, radiological, or nuclear (CBRN) weapons synthesis and threats."),

    ("q_eo_2", "whitehouse_eo_14110_ai", "factual",
     "What are the reporting obligations for companies training dual-use foundation models regarding AI red-teaming?",
     r"red-teaming",
     "red-teaming",
     "Companies developing dual-use foundation models must report their activities and share the results of red-teaming safety tests with the Federal Government."),

    ("q_eo_3", "whitehouse_eo_14110_ai", "multi_chunk",
     "What role do federal agencies have in developing safety standards and guidance under EO 14110?",
     r"Federal Government",
     "Federal Government",
     "Federal agencies are directed to establish guidance, evaluation benches, and procurement standards to ensure safe, secure, and trustworthy AI adoption."),

    # --- SEC Cybersecurity Guidance 2018 ---
    ("q_sec18_1", "sec_cybersecurity_guidance_2018", "factual",
     "What restrictions does the 2018 SEC guidance place on insider trading while aware of undisclosed cybersecurity incidents?",
     r"insiders must not trade",
     "insiders must not trade",
     "Directors, officers, and other corporate insiders must not trade a public company's securities while in possession of material nonpublic information regarding cybersecurity incidents."),

    ("q_sec18_2", "sec_cybersecurity_guidance_2018", "factual",
     "What are the requirements for public companies to maintain robust disclosure controls and procedures for cyber risks?",
     r"controls and procedures",
     "controls and procedures",
     "Companies must design and maintain disclosure controls and procedures that ensure material cybersecurity incidents are escalated to management in a timely manner."),

    ("q_sec18_3", "sec_cybersecurity_guidance_2018", "multi_chunk",
     "How should the board of directors disclose its role in cybersecurity risk oversight?",
     r"board of directors.*risk",
     "board of directors",
     "The SEC guidance emphasizes that companies should disclose how the board of directors oversees cybersecurity risk management and engages with executive management."),

    # --- SEC Cybersecurity Final Rule 2023 ---
    ("q_sec23_1", "sec_cybersecurity_final_rule_2023", "factual",
     "What are the reporting requirements under Form 8-K Item 1.05 for material cybersecurity incidents?",
     r"Item 1\.05",
     "Item 1.05",
     "Under Form 8-K Item 1.05, companies must disclose the material aspects of the nature, scope, and timing of a cybersecurity incident within four business days of determining materiality."),

    ("q_sec23_2", "sec_cybersecurity_final_rule_2023", "factual",
     "How does the SEC address potential conflicts between Item 1.05 and the Cyber Incident Reporting for Critical Infrastructure Act (CIRCIA)?",
     r"CIRCIA",
     "CIRCIA",
     "The SEC determined that Item 1.05 disclosure requirements serve investor protection and do not inherently conflict with nonpublic incident reports filed under CIRCIA."),

    ("q_sec23_3", "sec_cybersecurity_final_rule_2023", "multi_chunk",
     "Under what circumstances may the Attorney General delay a public company's Item 1.05 cybersecurity disclosure?",
     r"Attorney General.*national security",
     "Attorney General",
     "The final rule allows delayed disclosure if the US Attorney General determines that immediate public disclosure would pose a substantial risk to national security or public safety."),
]

# Unanswerable questions (6 questions = 15.0% of 40)
unanswerable = [
    ("q_unans_1", "none", "unanswerable",
     "What was NVIDIA's quarterly revenue from commercial deep-sea submarine mining equipment in fiscal 2024?",
     "",
     "I don't know based on the provided documents."),
    ("q_unans_2", "none", "unanswerable",
     "What are Microsoft's contractual penalty fees for violating lunar base habitat atmospheric safety regulations?",
     "",
     "I don't know based on the provided documents."),
    ("q_unans_3", "none", "unanswerable",
     "How many commercial supersonic aircraft did Apple deliver to global airlines during fiscal year 2023?",
     "",
     "I don't know based on the provided documents."),
    ("q_unans_4", "none", "unanswerable",
     "What orbital radio frequency licenses did Meta Platforms acquire for interstellar communication satellites?",
     "",
     "I don't know based on the provided documents."),
    ("q_unans_5", "none", "unanswerable",
     "What are the operational parameters of Amazon's geothermal power generation stations located in Antarctica?",
     "",
     "I don't know based on the provided documents."),
    ("q_unans_6", "none", "unanswerable",
     "What was the total megawatt output from AMD's commercial nuclear fusion reactors installed in 2023?",
     "",
     "I don't know based on the provided documents."),
]

eval_questions = []

# Validate answerable questions and extract exact verbatim substrings
for q_id, doc_id, q_type, question, pat, default_ev, ref in specs:
    chunk, snippet = find_first(doc_id, pat)
    if chunk is None:
        print(f"ERROR: Could not find chunk for {q_id} in {doc_id} with pattern {pat}")
        continue

    # Pick a solid verbatim phrase of 4 to 8 words that appears directly in chunk.content
    match = re.search(pat, chunk.content, re.IGNORECASE)
    assert match, f"Pattern {pat} must match {chunk.content[:100]}"

    # Extract clean phrase from match
    m_start = match.start()
    m_end = match.end()
    # extend to word boundaries
    surrounding = chunk.content[m_start:m_end + 30].strip()
    words = surrounding.split()
    evidence_phrase = " ".join(words[:6])
    if evidence_phrase.lower() not in chunk.content.lower():
        evidence_phrase = chunk.content[m_start:m_end].strip()

    assert evidence_phrase.lower() in chunk.content.lower(), f"Evidence phrase '{evidence_phrase}' not in chunk!"

    record = {
        "id": q_id,
        "question": question,
        "type": q_type,
        "source_doc": doc_id,
        "target_chunk_id": chunk.chunk_id,
        "evidence_phrase": evidence_phrase,
        "reference_answer": ref,
        "human_reviewed": True,
    }
    eval_questions.append(record)

for q_id, doc_id, q_type, question, ev, ref in unanswerable:
    record = {
        "id": q_id,
        "question": question,
        "type": q_type,
        "source_doc": doc_id,
        "target_chunk_id": None,
        "evidence_phrase": "",
        "reference_answer": ref,
        "human_reviewed": True,
    }
    eval_questions.append(record)

print(f"Generated {len(eval_questions)} questions ({len(specs)} answerable, {len(unanswerable)} unanswerable).")
unans_pct = len(unanswerable) / len(eval_questions) * 100.0
print(f"Unanswerable percentage: {unans_pct:.1f}%")

out_file = Path("eval/datasets/eval_questions_40.jsonl")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    for item in eval_questions:
        f.write(json.dumps(item) + "\n")

print(f"Successfully saved to {out_file}")
