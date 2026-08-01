from vectorstore import load_vectorstore
from llm import generate_answer, stream_answer
from config import RETRIEVE_K, GOOD_ABSOLUTE_SCORE,ABSOLUTE_CEILING, RELATIVE_RATIO


def sample_chunks_evenly(chunks, max_samples = 10):
    """Pick up to max_samples chunks spread evenly across the whole document, for summary-style questions."""
    if len(chunks) <= max_samples:
        return chunks
    step = len(chunks)/max_samples
    indices = [int(i * step) for i in range(max_samples)]
    return [chunks[i] for i in indices]

def is_summary_question(question):
    '''
    Ask the LLM whether this question is asking about the content of an
    uploaded document (even vaguely), as opposed to being unrelated/general.
    Used only when normal retrieval already failed to find anything relevant.
    '''
    classification_prompt = f'''
Answer with only "yes" or "no", nothing else.
Is the following question asking about the content, summary, or subject matter of an uploaded document,
rather than being a general/unrelated question?
Question: {question}
Answer :
'''
    try:
        response = generate_answer(classification_prompt)
        return response.strip().lower().startswith("yes")
    except Exception:
        # if classification fails for any reason, default to "not a summary question"
        # safer to fall through to general knowledge than to crash the whole answer
        return False
    

def filter_relevant_docs(retreived_results):
    """
        Decide which retrieved documents are actually relevant enough to use,
    based on a three-tier check rather than a single fixed threshold:
    1. If even the best result is too far away (above ABSOLUTE_CEILING), nothing is relevant.
    2. If the best result is good enough on its own merits (below GOOD_ABSOLUTE_SCORE),
       trust it regardless of how close the other results are - handles documents with
       multiple genuinely relevant, similarly-scored chunks.
    3. Otherwise, check if the best result clearly stands out from the second-best -
       a meaningful relative gap still counts as relevant even if not dramatically good.
    Returns a list of Document objects (empty list if nothing qualifies).
    """
    if not retreived_results:
        return []
    sorted_results = sorted(retreived_results, key=lambda pair: pair[1])
    best_doc, best_score = sorted_results[0]
    if best_score > ABSOLUTE_CEILING:
        return []
    if best_score < GOOD_ABSOLUTE_SCORE:
        return [doc for doc, score in sorted_results if score < GOOD_ABSOLUTE_SCORE]
    if len(sorted_results) > 1:
        second_best_score = sorted_results[1][1]
        if best_score < second_best_score * RELATIVE_RATIO:
            return [best_doc]
    return []

def _build_prompt_and_mode(question, vectorstore, chunks=None):
    """
    Shared retrieval + routing logic used by both the non-streaming and
    streaming answer functions. Returns (prompt, mode, sources).
    """
    if vectorstore is not None:
        retrevied_results = vectorstore.similarity_search_with_score(question, k=RETRIEVE_K)
    else:
        retrevied_results = []

    relevant_docs = filter_relevant_docs(retrevied_results)

    if relevant_docs:
        context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
        sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant_docs})
        prompt = f"""You are a helpful assistant with access to context from the user's uploaded documents.
Use the context below to answer the question if it is relevant. If the context only partially \
answers the question, use it for the part it covers and clearly supplement the rest with your own \
general knowledge, explicitly marking which part is which. If the context is not actually relevant \
to the question, ignore it and answer from general knowledge instead.

context from documents:
{context}
Question : {question}

Answer : """
        return prompt, "document", sources

    if chunks and is_summary_question(question):
        # normal retrieval found nothing relevant, but this looks like a genuine
        # vague/whole-document question - fall back to a broad sample instead of
        # incorrectly treating every failed retrieval as document-related
        sampled = sample_chunks_evenly(chunks)
        context = "\n\n---\n\n".join(doc.page_content for doc in sampled)
        sources = sorted({doc.metadata.get("source", "unknown") for doc in sampled})
        prompt = f"""You are a helpful assistant. The user asked a question about their uploaded document(s), \
but a normal search didn't find a specific, clearly matching passage. Below are representative excerpts \
sampled from across the whole document(s). Use them to answer as best you can. If the excerpts genuinely \
don't help answer the question, say so honestly and answer from general knowledge instead.

Sampled excerpts from the document(s):
{context}
Question : {question}

Answer : """
        return prompt, "document", sources

    sources = []
    prompt = f"""The user asked a question that is not covered by their uploaded documents. Answer it using your own general knowledge.
Question: {question}
Answer : """
    return prompt, "general", sources

def answer_question(question, vectorstore, history = None, chunks=None):
    """
    Retrieve relevant chunks for the question, decide whether they r actually relevant enough to use and generate an answer      accordingly.
    Returns (answer_text, mode, sources).
    """
    
    prompt, mode, sources = _build_prompt_and_mode(question, vectorstore, chunks=chunks)
    answer = generate_answer(prompt, history=history)
    return answer, mode, sources


def stream_answer_question(question, vectorstore,history=None, chunks = None):
    """
    Streaming version: retrieves and routes exactly like answer_question then yields the answer as it's generated instead of returning it whole.
     Yields tuples:
     ("meta", mode, sources) - exactly once, first, before any text
     ("chunk", text) - one per piece of generated text
    """
    prompt, mode, sources = _build_prompt_and_mode(question, vectorstore, chunks=chunks)
    yield("meta", mode, sources)
    for chunk in stream_answer(prompt, history=history, provider="openrouter"):
        yield ("chunk", chunk)