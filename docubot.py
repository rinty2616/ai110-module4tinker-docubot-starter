"""Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import glob
import os
import re


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory.
        self.documents = self.load_documents()  # List of (filename, text)

        # Build an inverted index so we know which files contain each word.
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """Load .md and .txt files from docs_folder."""
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Part 2)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """Map each lowercase word to the files where it appears."""
        index = {}

        for filename, text in documents:
            words = re.findall(r"[a-z0-9_]+", text.lower())
            for word in set(words):
                if word not in index:
                    index[word] = []
                index[word].append(filename)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Parts 2 and 3)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """Count meaningful query words that appear in text."""
        common_words = {
            "a", "an", "and", "are", "at", "be", "by", "can", "do",
            "does", "for", "from", "how", "i", "in", "is", "it", "me",
            "my", "of", "on", "or", "our", "the", "to", "was", "what",
            "when", "where", "which", "who", "why", "with", "you", "your",
        }
        query_words = set(re.findall(r"[a-z0-9_]+", query.lower()))
        query_words -= common_words
        text_words = set(re.findall(r"[a-z0-9_]+", text.lower()))
        return len(query_words & text_words)

    def retrieve(self, query, top_k=3):
        """Return the best matching (filename, snippet) pairs.

        Part 3: search paragraph-sized snippets rather than whole documents,
        and refuse matches with too little evidence.
        """
        results = []

        for filename, text in self.documents:
            # Split each document on blank lines into smaller chunks.
            paragraphs = re.split(r"\n\s*\n", text)
            for paragraph in paragraphs:
                snippet = paragraph.strip()
                if not snippet:
                    continue
                score = self.score_document(query, snippet)
                if score > 0:
                    results.append((score, filename, snippet))

        # Put the most relevant snippets first.
        results.sort(key=lambda result: result[0], reverse=True)

        # Refuse when no snippet matches meaningfully.
        meaningful_words = self.score_document(query, query)
        min_score = 1 if meaningful_words <= 1 else 2
        if not results or results[0][0] < min_score:
            return []

        return [(filename, snippet) for score, filename, snippet in results[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """Return snippets and filenames without involving an LLM."""
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """Ask Gemini to answer using only retrieved snippets."""
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """Join all documents for the naive generation baseline."""
        return "\n\n".join(text for _, text in self.documents)
