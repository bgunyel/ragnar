from io import BytesIO
from PIL import Image

from ollama import ListResponse, Client
from tqdm import tqdm


def check_and_pull_ollama_model(model_name: str, ollama_url: str) -> None:
    ollama_client = Client(host=ollama_url)
    response: ListResponse = ollama_client.list()
    available_model_names = [x.model for x in response.models]

    # Modified from https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    if model_name not in available_model_names:
        current_digest, bars = '', {}
        for progress in ollama_client.pull(model=model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest


def load_ollama_model(model_name: str, ollama_url: str) -> None:
    check_and_pull_ollama_model(model_name=model_name, ollama_url=ollama_url)
    ollama_client = Client(host=ollama_url)
    ollama_client.generate(model=model_name)  # Generate w/ prompt loads the model to memory


def get_flow_chart(rag_model):
    img_bytes = BytesIO(rag_model.graph.get_graph(xray=True).draw_mermaid_png())
    img = Image.open(img_bytes).convert("RGB")
    return img
