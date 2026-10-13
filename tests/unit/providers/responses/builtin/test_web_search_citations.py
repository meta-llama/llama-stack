# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.inline.responses.builtin.responses.types import CitationUrl
from ogx.providers.inline.responses.builtin.responses.utils import (
    _extract_citations_from_text,
    convert_chat_choice_to_response_message,
    extract_citations_from_text,
)
from ogx_api.inference import OpenAIChatCompletionResponseMessage, OpenAIChoice
from ogx_api.openai_responses import (
    OpenAIResponseAnnotationCitation,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseOutputMessageContentOutputText,
)


class TestExtractUrlCitationsFromText:
    """Regression tests for #6763: web_search never produced url_citation annotations."""

    def test_extracts_url_citation_with_span_since_previous_marker(self):
        text = "Transformers use self-attention <|url-1|>. They were introduced in 2017 <|url-2|>."
        citation_urls = {
            "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need"),
            "url-2": CitationUrl(url="https://en.wikipedia.org/wiki/Transformer", title="Transformer"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Transformers use self-attention. They were introduced in 2017."
        assert annotations == [
            OpenAIResponseAnnotationCitation(
                url="https://arxiv.org/abs/1706.03762",
                title="Attention Is All You Need",
                start_index=0,
                end_index=31,
            ),
            OpenAIResponseAnnotationCitation(
                url="https://en.wikipedia.org/wiki/Transformer",
                title="Transformer",
                start_index=31,
                end_index=61,
            ),
        ]
        assert cleaned_text[annotations[0].end_index] == "."
        assert cleaned_text[annotations[1].end_index] == "."

    def test_preserves_unknown_url_marker(self):
        text = "Some fact <|url-99|>."
        annotations, cleaned_text = _extract_citations_from_text(text, {}, {"url-1": CitationUrl(url="u", title="t")})

        assert annotations == []
        assert cleaned_text == "Some fact <|url-99|>."

    def test_file_and_url_markers_coexist_with_correct_indices(self):
        """A single turn can call both file_search and web_search; markers of both
        types must be strippable in one pass without corrupting each other's indices."""
        text = "From a doc <|file-abc123|>. From the web <|url-1|>."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = _extract_citations_from_text(text, citation_files, citation_urls)

        assert cleaned_text == "From a doc. From the web."
        assert annotations == [
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=10),
            OpenAIResponseAnnotationCitation(url="https://example.com", title="Example", start_index=10, end_index=24),
        ]

    def test_falls_back_to_whole_text_span_when_model_does_not_cite(self):
        text = "Transformers use self-attention."
        citation_urls = {
            "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
        }

        annotations, cleaned_text = extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == text
        assert len(annotations) == 1
        assert annotations[0].url == "https://arxiv.org/abs/1706.03762"
        assert annotations[0].start_index == 0
        assert annotations[0].end_index == len(text)

    def test_fallback_cites_only_first_source(self):
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
        }

        annotations, _ = extract_citations_from_text("Some text.", {}, citation_urls)

        assert len(annotations) == 1
        assert annotations[0].url == "https://a.com"

    def test_no_fallback_when_no_urls_were_retrieved(self):
        annotations, cleaned_text = extract_citations_from_text("No sources used here.", {}, {})

        assert annotations == []
        assert cleaned_text == "No sources used here."

    def test_real_marker_citation_takes_precedence_over_url_fallback(self):
        text = "Cited fact <|url-1|>. Uncited fact."
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
        }

        annotations, cleaned_text = extract_citations_from_text(text, {}, citation_urls)

        assert len(annotations) == 1
        assert annotations[0].url == "https://a.com"
        assert cleaned_text == "Cited fact. Uncited fact."

    def test_both_file_and_url_fallbacks_apply_independently(self):
        """When the model cites neither inline, both a file_citation and a url_citation
        fallback should be produced -- one per retrieval type, not just one overall."""
        text = "An answer that draws on both sources."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = extract_citations_from_text(text, citation_files, citation_urls)

        assert len(annotations) == 2
        assert isinstance(annotations[0], OpenAIResponseAnnotationFileCitation)
        assert isinstance(annotations[1], OpenAIResponseAnnotationCitation)
        assert annotations[0].index == len(cleaned_text)
        assert annotations[1].end_index == len(cleaned_text)


class TestConvertChatChoiceToResponseMessageWithUrlCitations:
    async def test_strips_url_marker_and_attaches_url_citation(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(content="Transformers use self-attention <|url-1|>."),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(
            choice,
            citation_urls={
                "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
            },
        )

        assert isinstance(result.content[0], OpenAIResponseOutputMessageContentOutputText)
        assert result.content[0].text == "Transformers use self-attention."
        assert len(result.content[0].annotations) == 1
        annotation = result.content[0].annotations[0]
        assert annotation.url == "https://arxiv.org/abs/1706.03762"
        assert annotation.title == "Attention Is All You Need"

    async def test_populates_fallback_url_citation_when_model_does_not_cite_inline(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(content="Transformers use self-attention."),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(
            choice,
            citation_urls={
                "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
            },
        )

        assert len(result.content[0].annotations) == 1
        assert result.content[0].annotations[0].url == "https://arxiv.org/abs/1706.03762"
