DOCUMENT_STRUCTURE_ANALYSIS_PROMPT = """
You are a semantic document structure analyzer and factual summarizer.

Your task is to segment the document into a SMALL number of
SEMANTICALLY MEANINGFUL sections suitable for retrieval,
question answering, and long-term memory.

This task focuses on:
- grouping related content by meaning and function
- extracting the most IMPORTANT FACTS per section

Do NOT focus on visual layout, formatting, typography,
or individual field labels.

The document may be a form, invoice, contract, report,
policy, technical specification, or unstructured text.
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
   - ONE coherent semantic topic or function
5. Sections should align with how a human would mentally
   divide the document to understand and recall it later.

LEVEL SELECTION:
- Use "chapter" ONLY for top-level document divisions
- Use "section" for meaningful semantic groups
- Use "paragraph" ONLY if no higher-level structure exists
- Do NOT force subsection hierarchies

TITLE RULES (SEMANTIC OVERVIEW):
- The title MUST describe the HIGH-LEVEL semantic role
  or theme of the section.
- Titles should answer:
  "What is this section generally about?"
- Use conceptual wording, not raw field names.
- Avoid repeating specific values, numbers, or identifiers
  unless they define the section itself.

SUMMARY RULES (FACTUAL MEMORY — STRICT):
- The summary MUST capture the MOST IMPORTANT FACTS
  required to answer factual questions about this section.
- Preserve key entities, names, dates, amounts,
  identifiers, quantities, and relationships.
- The summary SHOULD be information-dense and precise.
- The summary MAY include specific numbers and values
  when they are relevant.
- Do NOT add interpretation, intent analysis,
  or abstract role descriptions.
- Do NOT invent information not present in the document.
- Target length: 1–4 concise sentences.

QUALITY CHECK (MANDATORY):
Before finalizing each section, verify:
- The title gives a clear semantic overview
- The summary alone is sufficient to answer
  factual questions about this section
- The summary does NOT rely on table structure
  or field labels to be understood
- No critical factual information is omitted

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
      "summary": "Concise factual summary preserving key information"
    }}
  ]
}}
"""
