# 🧠 Agentic AI Assistant

An extensible AI assistant powered by large language models and tool integration. Built using **LangGraph** to support multi-step, memory-aware reasoning across tools like Spotify and beyond. This assistant also integrates **open-source frameworks** like **RealtimeTTS** and **RealtimeSTT** for speech-based interaction.

## ✨ Features

- 🗣️ **Conversational Interface**  
  Chat naturally with your assistant via speech (RealTimeTextToSpeech).

- 🎧 **Spotify Integration**  
  Ask it to play specific songs, artists, or playlists using the Spotify Web API.

- 🧠 **LangGraph Agent System**  
  Uses a graph-based control flow (LangGraph) for flexible, multi-step task execution.

- 🧠 **LLM-Powered Reasoning**  
  Uses OpenAI models for planning and decision-making.

- 🗃️ **Memory and Context Awareness**  
  Retains conversation history using memory nodes.

## 🛠️ Tech Stack

- **Python 3.11+**
- [LangGraph](https://github.com/langchain-ai/langgraph) – Graph-based agent orchestration
- [OpenAI API](https://platform.openai.com/) – LLM backend
- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) – Music playback
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) – Open-source speech synthesis
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) – Open-source speech recognition

## 🧩 Tools and Capabilities

| Tool        | Description                        |
|-------------|------------------------------------|
| Spotify     | Plays music via natural commands   |
| Web Search  | (Planned) Fetches live data        |
| Calendar    | (Planned) Schedules tasks          |
| Speech      | Real-time speech input/output (TextToSpeech + SpeechToText), with Open-Wake-Word integration |
