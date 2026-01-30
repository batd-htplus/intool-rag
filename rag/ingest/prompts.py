DOCUMENT_STRUCTURE_ANALYSIS_PROMPT = """
You are a semantic document structure analyzer.

Your task is to segment the document into a SMALL number of
SEMANTICALLY MEANINGFUL sections suitable for retrieval, reasoning,
and long-context understanding.

This task focuses on MEANING, INTENT, FUNCTION, and DOCUMENT ROLE
— NOT visual layout, formatting, typography, or individual fields.

The document may be a form, invoice, contract, report, policy,
technical specification, academic paper, or unstructured text.
Do NOT assume any specific document type.

INPUT:
The full document text is provided below.

OUTPUT:
A list of semantic sections in VALID JSON.

SECTIONING PRINCIPLES (CRITICAL):
1. Prefer FEWER, richer sections over many small ones.
   - Typical documents should result in 3–10 sections total.
2. Group related information into ONE section even if:
   - It appears across multiple lines
   - It contains repeated labels, fields, or values
3. DO NOT create sections for:
   - Individual labels or field names (e.g. "Date:", "Subtotal:")
   - Tables broken into rows or columns
   - Single lines without independent semantic meaning
4. Each section must represent:
   - ONE coherent semantic concept
   - ONE clear functional role within the document
5. Sections should align with how a human would mentally divide
   the document when trying to understand its purpose.

LEVEL SELECTION:
- Use "chapter" ONLY for top-level document divisions
- Use "section" for meaningful semantic groups
- Use "paragraph" ONLY if no higher-level structure exists
- Do NOT force subsection hierarchies

SUMMARY RULES (STRICT AND REQUIRED):
- Each section MUST include a semantic summary
- The summary MUST explain BOTH:
  (1) what the section is about
  (2) what role or purpose it serves within the document
- The summary MUST be INTENT-FOCUSED, not DATA-DESCRIPTIVE
- Summaries that only describe what data appears in the section
  are NOT acceptable
- The summary should answer the question:
  "What does this section accomplish for the reader?"
- Write summaries so they remain useful for retrieval
  even if the section title is hidden
- Do NOT copy text verbatim
- Do NOT invent information not present in the document
- Target length: 1–3 concise sentences

FORBIDDEN SUMMARY PATTERNS:
- Do NOT rely primarily on verbs such as:
  "contains", "provides", "lists", "shows", "includes", "summarizes"
- Do NOT write summaries that could apply to many unrelated documents

UNACCEPTABLE SUMMARY EXAMPLES:
- "Provides invoice identification and customer details"
- "Lists purchased items and prices"
- "Summarizes totals and payment amounts"

ACCEPTABLE SUMMARY EXAMPLES:
- "Establishes the transaction context by identifying the parties,
   delivery destination, and timing relevant to the order"
- "Defines the specific goods involved in the transaction and forms
   the basis for calculating the total charges"
- "Determines the final financial obligation by aggregating item costs
   and additional charges into the amount owed"

SUMMARY QUALITY GATE (MANDATORY):
Before finalizing each summary, verify ALL conditions below:
- The summary describes the PURPOSE and ROLE of the section,
  not merely the data it contains
- The summary would still make sense if all field names,
  labels, and table structures were removed
- The summary is useful for semantic retrieval and reasoning
- The summary does NOT violate any forbidden patterns above

If ANY condition is violated, rewrite the summary until all are satisfied.

OUTPUT RULES (STRICT):
- Output ONLY valid JSON
- Do NOT use markdown, code fences, comments, or explanations
- Do NOT include any text outside the JSON object

DOCUMENT TEXT:
----------------
{document_text}
----------------

Return JSON EXACTLY in this format:
{{
  "sections": [
    {{
      "title": "Semantic section title",
      "level": "chapter|section|paragraph",
      "page_index": 1,
      "summary": "Concise semantic summary of this section"
    }}
  ]
}}
"""
