import streamlit as st
import requests
import json
from openai import OpenAI
import google.generativeai as genai
from typing import Dict, Any, Optional
import io

# --- Constants & Default Values ---

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep your responses concise."

# Default user prompt template (leave empty if not needed)
DEFAULT_USER_TEMPLATE = ""

# --- Session State Initialization ---
# Initialize session state keys to ensure they exist on first run.
# This preserves values across app reruns (e.g., when the 'Send' button is clicked).

def init_session_state():
    """Initializes all necessary keys in Streamlit's session state."""
    defaults = {
        "provider": "OpenAI",
        "openai_api_key": "",
        "openai_model_name": "gpt-4o-mini",
        "google_api_key": "",
        "gemini_model_name": "gemini-2.5-pro",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model_name": "llama3",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_template": DEFAULT_USER_TEMPLATE,
        "user_input": "",
        "last_response": "",
        "last_provider_info": "",
        "pdf_extracted_text": "" # PDF 추출 결과를 저장할 키
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- API Call Logic ---

def call_openai(api_key: str, model: str, system_prompt: str, combined_user_prompt: str) -> str:
    """
    Calls the OpenAI Chat Completions API.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_user_prompt},
            ],
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return ""

def call_gemini(api_key: str, model: str, system_prompt: str, combined_user_prompt: str) -> str:
    """
    Calls the Google Gemini (Generative AI) API.
    """
    try:
        genai.configure(api_key=api_key)
        
        # System prompt handling for Gemini
        generation_config = {}
        safety_settings = {} # Add safety settings if needed
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Start a chat session to maintain context (though here we only send one message)
        # For a simple, non-chat use case, you can also use generate_content
        response = model_instance.generate_content(combined_user_prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Google Gemini API Error: {e}")
        return ""

def call_ollama(base_url: str, model: str, system_prompt: str, combined_user_prompt: str) -> str:
    """
    Calls a self-hosted Ollama API (assuming OpenAI-compatible /v1/chat/completions endpoint).
    """
    try:
        # Construct the API endpoint
        api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_user_prompt},
            ],
            "stream": False,  # As requested
        }
        
        response = requests.post(
            api_url, 
            json=payload, 
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        response_json = response.json()
        answer = response_json["choices"][0]["message"]["content"]
        return answer
    except requests.exceptions.ConnectionError:
        st.error(f"Ollama Connection Error: Could not connect to {api_url}. Is the server running?")
        return ""
    except requests.exceptions.Timeout:
        st.error("Ollama Error: Request timed out.")
        return ""
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
        return ""

def call_gemini_for_pdf_extraction(api_key: str, pdf_bytes: bytes) -> str:
    """
    Calls the Google Gemini API (gemini-2.5-pro) to extract text from a PDF.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Use a model that supports file inputs, like gemini-1.5-pro
        model_instance = genai.GenerativeModel(model_name="gemini-2.5-pro")
        
        # Create the PDF part for the prompt
        pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
        
        # Define the prompt for text extraction
        prompt = """
**System / Instruction (역할 지정)**
당신은 PDF 문서(스캔/텍스트 혼합 포함)에서 **정확한 OCR**을 수행하는 전문 분석가이다.
아래 규칙을 철저히 지켜 **지정된 출력 형식으로만** 내보낸다. 추측/보정/요약 금지. 원문 충실 재현.

**핵심 규칙**

1. **판독 불가/누락**은 생성하지 말고 `⟨UNREADABLE⟩`로 표기.
2. **원문 순서 보존:** 지면 상 좌→우, 상→하, **다단(2-column) 우선 규칙** 준수(1열 전체→2열 전체). 동시 배치 텍스트는 좌표/블록 순서로 정렬.
3. **줄바꿈/하이픈 처리:** 줄 끝 하이픈은 단어 연결(`hyphenation fix`), 문단 내 강제 줄바꿈 제거(단, 시/코드/주소 등은 유지).
4. **특수기호·수식·문자셋**: 손실 없이 복원. 수식은 LaTeX로 감싸기(`$…$` or `$$…$$`).
5. **표(Table)**: 각 표를 **TSV**와 **Markdown 표** 두 형태로 동시에 제공. 병합셀은 빈칸 유지 + `rowspan/colspan` 메모.
6. **체크박스/라디오/도형**: `☑/☐`, `●/○` 등으로 명시. 선택 해석 금지.
7. **각주/미주/주석**: 본문 위치에 각주 표식 유지, 하단에 `[Footnote n] …`로 모아 적기.
8. **머리말/꼬리말**: 페이지 번호/문서명은 본문과 분리해 `[Header]`, `[Footer]` 블록으로.
9. **이미지/도표/스탬프**: 내용 텍스트가 있으면 OCR, 없으면 **ALT 설명**으로 `![ALT: …]` 1줄.
10. **언어/코드 혼용**: 원어 유지. 날짜·숫자 **형태를 바꾸지 말 것**(정규화 금지).
11. **보안/민감정보**: 마스킹 흔적은 그대로 표기(예: `****`). 임의 복원 금지.
12. **무손실 원칙**: 해석/요약/치환 하지 말고 **보이는 그대로** 전사.

**출력 형식 (이 순서로만)**

```
=== OCR-METADATA ===
file_name: {{파일명}}
page_count: {{정수}}
mode: {{"auto" | "text-priority" | "layout-priority"}}
notes: [[선회/왜곡/저해상도/워터마크 등 감지된 이슈 간단 기재]]

=== PAGES ===
----- PAGE 1 -----
[Header]
{존재하면 헤더 텍스트 1~3줄}
[/Header]

[Body]
{본문 텍스트. 단락은 빈줄 1개로 구분. 다단은 1열 전체→2열 전체 순서}
- 표는 아래 규칙으로 삽입:
  [Table 1 - Markdown]
  | Col1 | Col2 | ...
  |------|------|---
  | ...  | ...  |
  [/Table 1]
  [Table 1 - TSV]
  Col1\tCol2\t...
  ...\t...\t...
  [/Table 1]
- 수식: $E=mc^2$
- 체크박스 예: ☑ 동의함 / ☐ 비동의
- 그림/도표: ![ALT: 바 차트(범례: A/B/C), 값 텍스트 없음]
[/Body]

[Footnotes]
[Footnote 1] …
[Footnote 2] …
[/Footnotes]

[Footer]
{존재하면 푸터 텍스트 1~3줄}
[/Footer]
----- PAGE 1 END -----

----- PAGE 2 -----
{동일 포맷 반복}
----- PAGE 2 END -----
```

**품질 제어(모델 내부 행동 지시)**

* 페이지별 **회전(0/90/180/270)** 자동 감지 후 재배치.
* 표는 선/격자 여부와 무관하게 셀 경계 추정. 숫자열은 숫자로 보존(쉼표/단위 유지, 변환 금지).
* **좌표 기반 블록 병합**으로 텍스트 순서 확정(도형/캡션은 본문 직후).
* **라틴/한글/기호** 안정 인식(ffi/fi 합자, ‘–’ vs ‘—’ 구분).
* **낮은 신뢰도** 토큰은 `⟨?⟩`로 둘러 표기(예: `개⟨?⟩발`).
* PDF에 **내장 텍스트**가 있으면 우선 추출하되, 손실/깨짐이 보이면 해당 블록만 이미지 OCR로 대체 후 병합.

**모드 스위치(선택, 기본: auto)**

* `text-priority`: 표 단순화, 글자 가독성 우선.
* `layout-priority`: 레이아웃 보존(캡션/박스/사이드바를 Body 내 블록으로 유지).

**입력**

* 첨부: `{{PDF 파일}}` (또는 페이지별 이미지 배열)
* 선택 매개변수:

  * `pages`: 예) `1-3,5,7-8`
  * `detect_language`: true
  * `output_tables`: true
  * `max_alt_length`: 30 (이미지 ALT 최대 글자수)

**출력 제한**

* 위의 **출력 형식 블록**만 출력. 다른 설명·사과·추가 코멘트 금지.

---

## 🔧 간단 사용 예

* **프롬프트**: 위 “범용 프롬프트” 그대로 붙여넣고

  * 첨부: `report.pdf`
  * 옵션: `pages=1-5`, `mode=layout-priority`, `output_tables=true`

* **모델 응답(요약 예시, 일부)**

```
=== OCR-METADATA ===
file_name: report.pdf
page_count: 12
mode: layout-priority
notes: [slight skew on page 2; watermarked background detected]

=== PAGES ===
----- PAGE 1 -----
[Header]
ACME Corp — Annual Summary (Confidential)
[/Header]

[Body]
서론
본 보고서는 …

[Table 1 - Markdown]
| 항목 | 값 | 단위 |
|-----|----|------|
| 길이 | 12.3 | cm |
[/Table 1]
[Table 1 - TSV]
항목\t값\t단위
길이\t12.3\tcm
[/Table 1]

그림 1. ![ALT: 선그래프(2019–2024, 6개 점, 범례 없음)]
[/Body]

[Footnotes]
[Footnote 1] 자료 출처: 내부 DB
[/Footnotes]

[Footer]
Page 1 of 12
[/Footer]
----- PAGE 1 END -----
```
        """
        
        # Generate content
        response = model_instance.generate_content([prompt, pdf_part])
        
        return response.text
    except Exception as e:
        st.error(f"Gemini PDF 추출 오류: {e}")
        return ""

# --- Helper Functions ---

def validate_inputs() -> bool:
    """
    Checks if the necessary API keys or configs for the selected provider are present.
    """
    provider = st.session_state.provider
    
    if provider == "OpenAI":
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API Key in the settings.")
            return False
        if not st.session_state.openai_model_name:
            st.error("Please enter an OpenAI model name in the settings.")
            return False
            
    elif provider == "Google Gemini":
        if not st.session_state.google_api_key:
            st.error("Please enter your Google Gemini API Key in the settings.")
            return False
        if not st.session_state.gemini_model_name:
            st.error("Please enter a Gemini model name in the settings.")
            return False

    elif provider == "Ollama (self-hosted)":
        if not st.session_state.ollama_base_url:
            st.error("Please enter the Ollama Base URL in the settings.")
            return False
        if not st.session_state.ollama_model_name:
            st.error("Please enter an Ollama model name in the settings.")
            return False
            
    if not st.session_state.user_input:
        st.warning("Please enter some text in the user input field.")
        return False
        
    return True

def get_llm_response():
    """
    Validates inputs and calls the appropriate LLM provider function.
    Updates session state with the response.
    """
    if not validate_inputs():
        return

    # Combine the user prompt template with the main user input
    # This is where the prompt components are assembled.
    combined_user_prompt = f"{st.session_state.user_template}\n\n{st.session_state.user_input}".strip()

    provider = st.session_state.provider
    response = ""
    provider_info = ""

    with st.spinner(f"Waiting for {provider} response..."):
        try:
            if provider == "OpenAI":
                provider_info = f"Provider: OpenAI | Model: {st.session_state.openai_model_name}"
                response = call_openai(
                    api_key=st.session_state.openai_api_key,
                    model=st.session_state.openai_model_name,
                    system_prompt=st.session_state.system_prompt,
                    combined_user_prompt=combined_user_prompt
                )
            elif provider == "Google Gemini":
                provider_info = f"Provider: Google Gemini | Model: {st.session_state.gemini_model_name}"
                response = call_gemini(
                    api_key=st.session_state.google_api_key,
                    model=st.session_state.gemini_model_name,
                    system_prompt=st.session_state.system_prompt,
                    combined_user_prompt=combined_user_prompt
                )
            elif provider == "Ollama (self-hosted)":
                provider_info = f"Provider: Ollama | Model: {st.session_state.ollama_model_name}"
                response = call_ollama(
                    base_url=st.session_state.ollama_base_url,
                    model=st.session_state.ollama_model_name,
                    system_prompt=st.session_state.system_prompt,
                    combined_user_prompt=combined_user_prompt
                )
        except Exception as e:
            # This is a fallback catch-all, though specific errors are handled in provider functions.
            st.error(f"An unexpected error occurred: {e}")
    
    # Store the response in session state so it persists
    st.session_state.last_response = response
    st.session_state.last_provider_info = provider_info

# --- PDF Extraction Logic ---
def handle_pdf_extraction():
    """
    Handles the logic for PDF extraction when the button is clicked.
    """
    uploaded_file = st.session_state.get("pdf_file")
    
    if not uploaded_file:
        st.warning("PDF 파일을 먼저 업로드하세요.")
        return

    if not st.session_state.google_api_key:
        st.error("PDF 추출을 위해 Google Gemini API 키가 필요합니다. 사이드바 'Settings'에서 키를 입력해주세요.")
        return

    with st.spinner("PDF에서 텍스트를 추출 중... (Gemini API 사용)"):
        pdf_bytes = uploaded_file.getvalue()
        extracted_text = call_gemini_for_pdf_extraction(
            api_key=st.session_state.google_api_key,
            pdf_bytes=pdf_bytes
        )
        st.session_state.pdf_extracted_text = extracted_text

# --- Streamlit UI ---

def build_ui():
    """Constructs the Streamlit UI components."""
    
    # --- Settings Sidebar ---
    # All settings are placed in the sidebar to keep the main UI clean.
    with st.sidebar:
        st.title("Settings")
        st.write("Configure your LLM provider and prompts here.")

        # --- Model Provider Selection ---
        st.selectbox(
            "Select LLM Provider",
            options=["OpenAI", "Google Gemini", "Ollama (self-hosted)"],
            key="provider"
        )
        
        st.divider()

        # --- Provider-Specific Settings ---
        provider = st.session_state.provider
        
        if provider == "OpenAI":
            st.subheader("OpenAI Settings")
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                help="Get your key from https://platform.openai.com/api-keys"
            )
            st.text_input(
                "Model Name",
                key="openai_model_name",
                help="E.g., gpt-4o-mini, gpt-4-turbo"
            )

        elif provider == "Google Gemini":
            st.subheader("Google Gemini Settings")
            st.text_input(
                "Google Gemini API Key",
                type="password",
                key="google_api_key",
                help="Get your key from https://aistudio.google.com/app/api-keys"
            )
            st.text_input(
                "Model Name",
                key="gemini_model_name",
                help="E.g., gemini-2.5-pro"
            )

        elif provider == "Ollama (self-hosted)":
            st.subheader("Ollama (self-hosted) Settings")
            st.text_input(
                "Ollama Base URL",
                key="ollama_base_url",
                help="E.g., http://localhost:11434"
            )
            st.text_input(
                "Model Name",
                key="ollama_model_name",
                help="E.g., llama3, phi3 (must be compatible with OpenAI API endpoint)"
            )
            st.caption("Note: Assumes Ollama is serving an OpenAI-compatible API at `/v1/chat/completions`.")

        st.divider()

        # --- Prompt Configuration ---
        st.subheader("Prompt Configuration")
        st.text_area(
            "System Prompt",
            key="system_prompt",
            height=150,
            help="The system-level instructions for the LLM."
        )
        st.text_area(
            "User Prompt Template",
            key="user_template",
            height=100,
            help="Optional text to prepend to your main input. (e.g., 'Summarize the following text: ')"
        )

    # --- Main UI ---
    st.title("LLM Playground")

    # --- Tabs for different modes ---
    tab1, tab2 = st.tabs(["LLM Playground", "PDF 텍스트 추출"])

    # --- Tab 1: LLM Playground (Original) ---
    with tab1:
        # Main user input text area
        st.text_area(
            "User Input",
            key="user_input",
            height=300,
            placeholder="Enter your prompt here..."
        )

        # Send button
        st.button(
            "Send",
            on_click=get_llm_response, # Function to call when clicked
            type="primary",
            key="send_button"
        )

        # --- Response Area ---
        if st.session_state.last_response:
            st.markdown("---")
            st.info(st.session_state.last_provider_info)
            # Use st.container with a border for a nicely formatted box
            with st.container(border=True):
                st.markdown(st.session_state.last_response)
    
    # --- Tab 2: PDF Text Extraction ---
    with tab2:
        st.header("PDF 텍스트 추출 (Gemini API)")
        st.info(
            "이 기능은 Google Gemini API ('gemini-2.5-pro')를 사용합니다. "
            "사이드바 'Settings'에서 Google Gemini API 키가 올바르게 설정되었는지 확인하세요."
        )

        # PDF file uploader
        st.file_uploader(
            "PDF 파일을 업로드하세요",
            type=["pdf"],
            key="pdf_file"
        )

        # Extraction button
        st.button(
            "텍스트 추출하기",
            on_click=handle_pdf_extraction,
            key="pdf_extract_button"
        )

        # --- PDF Extraction Response Area ---
        if st.session_state.pdf_extracted_text:
            st.markdown("---")
            st.subheader("추출된 텍스트")
            with st.container(border=True, height=500):
                st.markdown(st.session_state.pdf_extracted_text)


# --- App Entry Point ---

def main():
    st.set_page_config(
        page_title="LLM Playground",
        page_icon="🤖",
        layout="centered"
    )
    
    # 1. Initialize session state
    init_session_state()
    
    # 2. Build the UI
    build_ui()

if __name__ == "__main__":
    main()