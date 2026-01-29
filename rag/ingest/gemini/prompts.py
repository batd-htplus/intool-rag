DOCUMENT_STRUCTURE_ANALYSIS_PROMPT = """
You are a semantic document structure analyzer.

Your task is to segment the document into a SMALL number of
SEMANTICALLY MEANINGFUL sections suitable for retrieval, reasoning,
and long-context understanding.

This task focuses on MEANING and INTENT — NOT visual layout,
formatting, or individual fields.

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
4. Each section must represent ONE coherent semantic concept
   (e.g. overview, parties, items, terms, scope, conclusions).

LEVEL SELECTION:
- Use "chapter" ONLY for top-level document divisions
- Use "section" for meaningful semantic groups
- Use "paragraph" ONLY if no higher-level structure exists
- Do NOT force subsection hierarchies

SUMMARY RULES (REQUIRED):
- Each section MUST include a semantic summary
- The summary should abstract and describe the meaning of the section
- Do NOT copy text verbatim
- Do NOT invent information not present in the document
- Target length: 1–3 concise sentences

OUTPUT RULES (STRICT):
- Output ONLY valid JSON
- Do NOT use markdown, code fences, or explanations
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
""".strip()
