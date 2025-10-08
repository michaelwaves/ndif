from fastapi import FastAPI
import nnsight
import uvicorn
from dataclasses import dataclass, field
import nnsight


app = FastAPI(title="NDIF Completions API", version="1.0.0")

@dataclass
class InferenceRequest:
    model:str ="tiiuae/Falcon3-7B-Base"
    prompt:str = "The CN Tower is located in"
    cache_layers:list[int] = field(default_factory=lambda:[1,2,3])
    interventions: dict= field(default_factory=dict)
    max_tokens = 1
    temperature = 1
    return_logits=True

request = InferenceRequest()
def prepare_nnsight_config():
    nnsight.CONFIG.set_default_api_key("api key")
    #nnsight.CONFIG.API.HOST = "172.26.64.1:5001"
    nnsight.CONFIG.API.HOST = "localhost:5001"
    nnsight.CONFIG.API.SSL = False


@app.post("/completions")
def get_completion():
    prepare_nnsight_config()

    layers = set()
    layers.update(request.cache_layers)
    layers = sorted(list(layers))

    model = nnsight.LanguageModel(request.model)
    with model.generate(request.prompt, max_new_tokens = request.max_tokens, remote=True) as tracer:
        # all_indices = list().save()
        all_sampled_tokens = list().save()
        all_layer_activations = list().save()
        all_logits =list().save()

        with tracer.all():
            layer_activations = {layer: list().save() for layer in layers}
            logits = list().save()
            sampled_tokens = list().save()

            for layer in layers:
                hidden_states = model.model.layers[layer].output[0]
                print(f"Hidden States: {hidden_states.shape}")
                residual = model.model.layers[layer].output[1]
                print(f"Residual: {residual.shape}")
                total_layer_output = hidden_states+residual
                layer_activations[layer].append(total_layer_output.cpu())
            if request.return_logits:
                logits.append(model.logits.output.log_softmax(dim=-1).cpu())

            output = model.samples.output.item()
            sampled_tokens.append(output)

            all_sampled_tokens.append(sampled_tokens)
            all_layer_activations.append(layer_activations)
            all_logits.append(logits)


        output = model.output.save()
    
    print(output)
    print(output.keys())
    print(output["logits"].shape)
    return "hello"


@app.get("/v1/models")
async def list_models():
    """
    List available models (placeholder).
    """
    return {
        "object": "list",
        "data": [
            {"id": "openai-community/gpt2", "object": "model", "owned_by": "openai-community"},
            {"id": "tiiuae/Falcon3-7B-Base", "object": "model", "owned_by": "tiiuae"},
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
