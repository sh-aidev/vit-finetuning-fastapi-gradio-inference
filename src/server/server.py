from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gradio as gr
import sys
from src.utils.logger import logger
from src.utils.config import Config
from src.core.inference import VITInferenceHF
import json, io
from PIL import Image
from typing import Annotated, Tuple
from fastapi import APIRouter, File

def get_router(cfg: Config) -> Tuple[APIRouter, VITInferenceHF]:
    v1Router = APIRouter()
    classifier = VITInferenceHF(cfg)
    @v1Router.post("/classifier", status_code=200)
    def classify(
        file: Annotated[bytes, File()]
        ) -> dict:
        img = Image.open(io.BytesIO(file))
        img = img.convert("RGB")
        return classifier.run(img)

    @v1Router.get("/health")
    def health():
        return {"message": "ok"}
    return v1Router, classifier

class VITServer:
    def __init__(self, cfg: Config) -> None:
        self.config = cfg
        logger.debug("Configs Loaded...")

        self.server = FastAPI()
        logger.debug("FastAPI server initialized...")
        self.server.add_middleware(
            CORSMiddleware,
            allow_origins = ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            
        )
        logger.debug("CORS Middleware added...")
        v1Router, classifier = get_router(self.config)
        logger.debug("Router and classifier initialized...")
        self.server.include_router(v1Router, prefix="/v1")
        logger.debug("Router added to server...")

        self.classifier = classifier
        logger.debug("Classifier initialized...")

    def fast_api_serve(self) -> None:
        logger.debug(f"Starting FastAPI server at {self.config.vit_config.server.host}:{self.config.vit_config.server.port}")
        uvicorn.run(self.server, port=self.config.vit_config.server.port, host=self.config.vit_config.server.host)
    

    def gradio_server(self):
        logger.debug("Starting Gradio server...")
        im = gr.Image(type="pil", label="Input Image")
        demo = gr.Interface(
            fn=self.classifier.run,
            inputs=[im],
            outputs=[gr.Label(num_top_classes = 10)],
        )
        try:
            demo.launch(server_name = self.config.vit_config.server.host, server_port = self.config.vit_config.server.port)
        except KeyboardInterrupt:
            print("\n")
            logger.error("Keyboard Interrupted...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(0)