from app.config import settings
from app.db.vector_store import DocumentChunk
from app.utils.logger import logger
from app.utils.errors import LLMError

_client = None

SYSTEM_PROMPT = (
    "Ban la tro ly ky thuat nuoi tom chuyen nghiep cua he thong SmartShrimp. "
    "Tra loi cac cau hoi ky thuat ve nuoi tom dua tren tai lieu tham khao duoc cung cap. "
    "Chi dung thong tin tu tai lieu. Tra loi bang tieng Viet, ro rang va chuyen nghiep. "
    "Neu tai lieu khong du thong tin, noi khong tim thay. Khong tu bia dat thong tin ngoai tai lieu."
)


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_context(chunks: list[DocumentChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Tai lieu {i} - {chunk.source_file}, phan {chunk.chunk_index}]\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    question: str,
    chunks: list[DocumentChunk],
) -> tuple[str, str]:
    """
    Generate an answer using Gemini LLM given retrieved context chunks.

    Returns:
        Tuple of (answer_text, model_version_string)

    Raises:
        LLMError: If the Gemini API call fails
    """
    context = _build_context(chunks)
    user_prompt = (
        f"Tai lieu tham khao:\n\n{context}\n\n"
        f"---\n\n"
        f"Cau hoi: {question}\n\n"
        f"Tra loi:"
    )

    try:
        from google.genai import types as genai_types
        client = _get_client()
        response = client.models.generate_content(
            model=settings.gemini_llm_model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        answer = response.text.strip()
        logger.debug(f"Generated answer ({len(answer)} chars)")
        return answer, settings.gemini_llm_model

    except LLMError:
        raise
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise LLMError(f"Gemini generation error: {str(e)}")