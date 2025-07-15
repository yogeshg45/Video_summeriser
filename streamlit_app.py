# ✅ Set Hugging Face cache paths early to avoid PermissionError
import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/huggingface/sentence_transformers"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["SENTENCE_TRANSFORMERS_HOME"], exist_ok=True)

# ✅ Now import remaining libraries
import csv
import torch
import pickle
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

# Constants
SUMMARY_FOLDER = "src/foldersumarry"
VIDEO_CSV = "src/videoo.csv"
EMBEDDING_CACHE = "src/summary_embeddings.pkl"
SEARCH_LOG = "/tmp/search_history.csv"

# Load summaries
def load_all_summaries(folder_path):
    summaries, file_names = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    summaries.append(text)
                    file_names.append(filename)
    return summaries, file_names

# Load video info
def load_video_info(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines='warn')
    return {row["filename"]: row.to_dict() for _, row in df.iterrows()}

@st.cache_resource
def load_embeddings():
    if os.path.exists(EMBEDDING_CACHE):
        with open(EMBEDDING_CACHE, "rb") as f:
            cache = pickle.load(f)
        return cache["summaries"], cache["filenames"], cache["embeddings"]
    else:
        summaries, file_names = load_all_summaries(SUMMARY_FOLDER)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(summaries, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        with open(EMBEDDING_CACHE, "wb") as f:
            pickle.dump({"summaries": summaries, "filenames": file_names, "embeddings": embeddings}, f)
        return summaries, file_names, embeddings

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    device = 0 if torch.cuda.is_available() else -1
    llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return embed_model, llm_pipeline

def build_faiss_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def search(index, query_vector, top_k=3):
    query_vector = np.array(query_vector, dtype=np.float32)
    if query_vector.ndim == 1:
        query_vector = query_vector[np.newaxis, :]
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]

def generate_answer(llm_pipeline, query, context, max_length=150):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = llm_pipeline(prompt, max_length=max_length, do_sample=False)
    return result[0]['generated_text'].strip()

def generate_follow_up_questions(llm_pipeline, context):
    prompt = f"Context:\n{context}\n\nGenerate 2 short conceptual questions to test understanding:"
    result = llm_pipeline(prompt, max_length=100, do_sample=False)[0]['generated_text']
    return [q.strip().lstrip("0123456789. ") for q in result.split("\n") if q.strip()][:2]

def extract_score(text):
    match = re.search(r'\b([1-5])\b', text)
    return int(match.group(1)) if match else 0

def log_search(query, match_data):
    file_exists = os.path.exists(SEARCH_LOG)
    with open(SEARCH_LOG, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=match_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(match_data)

# Streamlit App
st.title("🎓 Video Summary Q&A")
query = st.text_input("🔍 Ask your question here:")

# Session state initialization
defaults = {
    "step": 0,
    "answers": [],
    "questions": [],
    "current_question_index": 0,
    "video_files": [],
    "contexts": [],
    "scores": [],
    "matched": [],
    "final_answer": "",
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

if query and st.session_state.step == 0:
    summaries, file_names, embeddings = load_embeddings()
    embed_model, llm_pipeline = load_models()
    video_info_map = load_video_info(VIDEO_CSV)
    index = build_faiss_index(embeddings)

    query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = search(index, query_vec)

    combined_context = ""
    st.session_state.contexts.clear()
    st.session_state.video_files.clear()
    st.session_state.matched.clear()

    st.markdown("### 🔍 Top matched videos:")
    for rank, i in enumerate(idxs, 1):
        fname = file_names[i]
        vinfo = video_info_map.get(fname, {})
        summary = summaries[i]
        st.markdown(f"**{rank}. {fname}**  \n"
                    f"- **Topic:** {vinfo.get('topic', 'Unknown')}  \n"
                    f"- **Subtopic:** {vinfo.get('subtopic', 'Unknown')}  \n"
                    f"- **Concept:** {vinfo.get('Concept', 'Unknown')}  \n"
                    f"- **Video URL:** [{vinfo.get('video_url', '')}]({vinfo.get('video_url', '')})")
        combined_context += summary + " "
        st.session_state.contexts.append(summary)
        st.session_state.video_files.append(fname)
        st.session_state.matched.append(summary)

        log_search(query, {
            "user_question": query,
            "matched_file": fname,
            "topic": vinfo.get('topic', ''),
            "subtopic": vinfo.get('subtopic', ''),
            "Concept": vinfo.get('Concept', ''),
            "video_url": vinfo.get('video_url', '')
        })

    st.markdown("### ✅ Final Answer")
    if not st.session_state.final_answer:
        st.session_state.final_answer = generate_answer(llm_pipeline, query, combined_context)
    st.success(st.session_state.final_answer)

    if st.button("Proceed to Follow-up Questions", key="proceed_questions"):
        st.session_state.questions.clear()
        for context in st.session_state.contexts:
            st.session_state.questions.extend(generate_follow_up_questions(llm_pipeline, context))
        st.session_state.step = 1
        st.rerun()

elif st.session_state.step == 1:
    i = st.session_state.current_question_index
    if i < len(st.session_state.questions):
        q = st.session_state.questions[i]
        with st.form(key=f"form_{i}"):
            user_input = st.text_input(f"🧠 Q{i+1}: {q}", key=f"q_{i}")
            submitted = st.form_submit_button("Submit Answer")
            if submitted:
                if user_input.strip() == "":
                    st.warning("Answer cannot be empty.")
                else:
                    st.session_state.answers.append(user_input.strip())
                    st.session_state.current_question_index += 1
                    if st.session_state.current_question_index == len(st.session_state.questions):
                        st.session_state.step = 2
                    st.rerun()
    else:
        st.success("✅ All questions answered!")
        if st.button("🎯 View Best Video and Feedback"):
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    embed_model, llm_pipeline = load_models()
    st.markdown("### 🔍 Evaluating your answers...")
    scores = [0] * len(st.session_state.video_files)

    for i, user_ans in enumerate(st.session_state.answers):
        context = st.session_state.contexts[i // 2]
        question = st.session_state.questions[i]
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"User's Answer: {user_ans}\n"
            f"Score (1 to 5) and explain why:")
        with st.spinner(f"Scoring Answer {i+1}..."):
            result = llm_pipeline(prompt, max_length=100, do_sample=False)[0]['generated_text']
        score = extract_score(result)
        scores[i // 2] += score
        st.markdown(f"**Answer {i+1} Feedback:** {result}")

    best_index = np.argmax(scores)
    best_file = st.session_state.video_files[best_index]
    best_info = load_video_info(VIDEO_CSV).get(best_file, {})

    st.markdown("### 🎬 Best Matched Video for You")
    st.markdown(f"**Filename:** {best_file}  \n"
                f"- **Topic:** {best_info.get('topic', 'Unknown')}  \n"
                f"- **Subtopic:** {best_info.get('subtopic', 'Unknown')}  \n"
                f"- **Concept:** {best_info.get('Concept', 'Unknown')}  \n"
                f"- **Video URL:** [{best_info.get('video_url', '')}]({best_info.get('video_url', '')})")

    if st.button("🔁 Restart"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
