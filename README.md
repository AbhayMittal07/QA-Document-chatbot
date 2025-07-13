# 📄 Document QA Chatbot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-brightgreen)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent **Document Question Answering Chatbot** built using **LangChain**, **LlamaIndex**, **Metaphor API**, and **Streamlit**. This system can answer user queries by first searching uploaded documents, then falling back to a language model, and finally performing internet search if necessary.

---

## 🧠 Overview

**Document QA Chatbot** is a multi-stage **Retrieval-Augmented Generation (RAG)** pipeline that intelligently retrieves context-aware answers:

- ✅ Queries uploaded PDFs, Word, and Text documents
- 🤖 Uses RAG with LLMs for generative answers
- 🌐 Searches the internet using the **Metaphor API** if needed
- 💡 Streamlit-powered UI for seamless user interaction

---

## 📌 Features

- 📁 Upload and index documents of various formats (PDF, DOCX, TXT)
- 🔍 Retrieve precise answers from documents using vector embeddings
- 🧠 Generate fallback answers using a language model
- 🌐 Search the web for out-of-context queries using Metaphor API
- 🖥️ Interactive web UI built with Streamlit

---

## 🏗️ Architecture

```text
User Query
   │
   ▼
[1. Search Indexed Documents via LlamaIndex]
   │      └── Answer Found → Return Result
   ▼
[2. Fallback to RAG LLM using LangChain]
   │      └── Answer Generated → Return Result
   ▼
[3. External Search via Metaphor API]
   │      └── Final Attempt → Search Web + Summarize Result
   ▼
[Display Answer + Document Source / Web Link]
