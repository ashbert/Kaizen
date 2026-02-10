"""
Modal vLLM deployment for Kaizen.

Deploys a vLLM-powered OpenAI-compatible inference endpoint on Modal.
The endpoint serves instruct-tuned models for structured code generation.

Usage:
    # Deploy
    modal deploy modal_vllm.py

    # Test locally
    modal run modal_vllm.py

    # Call the endpoint
    curl -X POST https://<your-app>.modal.run/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen2.5-Coder-32B-Instruct",
             "messages": [{"role": "user", "content": "Hello"}]}'

Requirements:
    pip install "kaizen[modal]"
    modal token set
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
GPU = "A100-80GB:1"

app = modal.App("kaizen-vllm")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "torch>=2.4.0")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.cls(
    image=vllm_image,
    gpu=GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    timeout=900,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
class Inference:
    """vLLM inference server as a Modal class."""

    model_id: str = MODEL_ID

    @modal.enter()
    def load_model(self) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=32768,
            gpu_memory_utilization=0.90,
        )

    @modal.method()
    def generate(self, messages: list[dict], max_tokens: int = 4096) -> dict:
        """Generate a chat completion."""
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=0.1,
            max_tokens=max_tokens,
        )

        result = self.llm.chat(
            messages=[messages],
            sampling_params=sampling,
        )
        output = result[0]
        text = output.outputs[0].text

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "model": self.model_id,
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            },
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def v1_chat_completions(self, request: dict) -> dict:
        """OpenAI-compatible /v1/chat/completions endpoint."""
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 4096)
        return self.generate.local(messages, max_tokens=max_tokens)


@app.local_entrypoint()
def main():
    """Quick smoke test."""
    inference = Inference()
    result = inference.generate.remote(
        messages=[
            {"role": "system", "content": "Reply concisely."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    )
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"Usage: {result['usage']}")
