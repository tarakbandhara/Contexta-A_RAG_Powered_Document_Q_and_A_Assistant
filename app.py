import uuid
import asyncio
import chainlit as cl
from ingestion import load_file, chunk_documents
from vectorstore import build_vectorstore, load_vectorstore
from rag_chain import stream_answer_question
import os

FILE_TYPE_ICONS = {
    ".pdf": "📕",
    ".docx": "📘",
    ".txt": "📄",
    ".csv": "📊"
}

def get_file_icon(filename):
    ext = os.path.splitext(filename)[1].lower()
    return FILE_TYPE_ICONS.get(ext,"📄") # default fallback icon


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What can you do?",
            message="What can you do?",
        ),
        cl.Starter(
            label="How do I ask about a document?",
            message="How do I upload and ask questions about a document?",
        ),
        cl.Starter(
            label="Tell me something interesting",
            message="Tell me something interesting.",
        ),
    ]

@cl.on_chat_start
async def start():
    cl.user_session.set("vectorstore", None)
    cl.user_session.set("history", [])
    cl.user_session.set("chunks", None)
    cl.user_session.set("collection_name", f"session_{uuid.uuid4().hex}")

@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        processing_msg = cl.Message(content="Reading and processing your document(s)...")
        await processing_msg.send()
        try:
            all_docs = []
            for element in message.elements:
                docs = load_file(element.path)
                for doc in docs:
                    doc.metadata["source"] = element.name
                all_docs.extend(docs)
            chunks = chunk_documents(all_docs)
            collection_name = cl.user_session.get("collection_name")
            vs = build_vectorstore(chunks, collection_name=collection_name)
            cl.user_session.set("vectorstore", vs)
            cl.user_session.set("chunks", chunks)

            processing_msg.content = f"Got it! Inserted {len(message.elements)} file(s)."
            await processing_msg.update()

            if not message.content.strip():
                return
        except Exception as e:
            processing_msg.content = f"Something went wrong while processing your file(s): {e}"
            await processing_msg.update()
            return
    
    vectorstore = cl.user_session.get("vectorstore")
    history = cl.user_session.get("history")
    chunks = cl.user_session.get("chunks")

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()
    msg = None

    mode = None
    sources = []
    full_answer = ""
    first_chunk_recevied = False

    try:
        for item in stream_answer_question(message.content, vectorstore, history=history, chunks=chunks):
            if item[0] == "meta":
             _, mode, sources = item
            elif item[0] == 'status':
                _, status_text = item
                await cl.Message(content=status_text).send()
            else:
                _, text = item
                if not first_chunk_recevied:
                    await thinking_msg.remove()
                    msg = cl.Message(content="")
                    await msg.send()
                    first_chunk_recevied = True
                full_answer += text
                await msg.stream_token(text)
                await asyncio.sleep(0.0375)
    except Exception as e:
        await thinking_msg.remove()
        error_msg = cl.Message(content = f"sorry, something went wrong while generating a response : {e}")
        await error_msg.send()
        return
    if mode == "document":
        labeled_sources = [f"{get_file_icon(src)} {src}" for src in sources]
        badge = f"\n\n📄 *From: {', '.join(labeled_sources)}*"
    else:
        badge = "\n\n🧠 *General Knowledge*"
    msg.content = full_answer + badge
    await msg.update()

    history.append({'role':'user', 'content': message.content})
    history.append({'role':'assistant', 'content':full_answer})
    cl.user_session.set("history", history)


@cl.on_chat_end
async def end():
    collection_name = cl.user_session.get("collection_name")
    if collection_name:
        try:
            vs = load_vectorstore(collection_name = collection_name)
            vs.delete_collection()
        except Exception:
            pass


