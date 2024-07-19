import asyncio
import signal

import fire

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from omegaconf import OmegaConf

from toolchain.utils import get_config_dir, parse_config
from .api.config import ModelInferenceHydraConfig
from .api.endpoints import ChatCompletionRequest, ChatCompletionResponseStreamChunk

from .api_instance import get_inference_api_instance


load_dotenv()


GLOBAL_CONFIG = None


def get_config():
    return GLOBAL_CONFIG


def handle_sigint(*args, **kwargs):
    print("SIGINT or CTRL-C detected. Exiting gracefully", args)
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.stop()


app = FastAPI()


@app.on_event("startup")
async def startup():
    global InferenceApiInstance

    config = get_config()
    hydra_config = ModelInferenceHydraConfig(
        **OmegaConf.to_container(config["model_inference_config"], resolve=True)
    )
    model_inference_config = hydra_config.convert_to_model_inferene_config()

    InferenceApiInstance = await get_inference_api_instance(
        model_inference_config,
    )
    await InferenceApiInstance.initialize()


@app.on_event("shutdown")
async def shutdown():
    global InferenceApiInstance

    print("shutting down")
    await InferenceApiInstance.shutdown()


# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
semaphore = asyncio.Semaphore(1)


@app.post(
    "/inference/chat_completion", response_model=ChatCompletionResponseStreamChunk
)
def chat_completion(request: Request, exec_request: ChatCompletionRequest):
    if semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail="Only a single concurrent request allowed right now.",
        )

    async def sse_generator(event_gen):
        try:
            async for event in event_gen:
                yield f"data: {event.json()}\n\n"
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("Generator cancelled")
            await event_gen.aclose()
        finally:
            semaphore.release()

    async def event_gen():
        async for event in InferenceApiInstance.chat_completion(exec_request):
            yield event

    return StreamingResponse(
        sse_generator(event_gen()),
        media_type="text/event-stream",
    )


def main(config_path: str, port: int = 5000, disable_ipv6: bool = False):
    global GLOBAL_CONFIG
    config_dir = get_config_dir()
    GLOBAL_CONFIG = parse_config(config_dir, config_path)

    signal.signal(signal.SIGINT, handle_sigint)

    import uvicorn

    # FYI this does not do hot-reloads
    listen_host = "::" if not disable_ipv6 else "0.0.0.0"
    print(f"Listening on {listen_host}:{port}")
    uvicorn.run(app, host=listen_host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
