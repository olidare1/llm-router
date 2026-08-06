import asyncio
import logging
import time
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==============================================================================
# 1. SETUP & KONFIGURATION
# ==============================================================================
# Logging konfigurieren, damit wir im Terminal nachvollziehen können,
# wie Anfragen durch das System fließen (Routing-Entscheidungen, Modellwahl, Latenz).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("llm-router")

# FastAPI Instanz mit Metadaten initialisieren (wichtig für automatische Swagger-UI / docs)
app = FastAPI(
    title="Intelligenter LLM-Router (Model Multiplexer)",
    description=(
        "Ein API-Gateway / Router, der eingehende Prompts analysiert und "
        "dynamisch zwischen schnellen/günstigen und leistungsstarken/teuren Modell-APIs entscheidet."
    ),
    version="1.0.0",
)


class ModelType(str, Enum):
    FAST = "llama-3-8b-instruct"
    HEAVY = "gpt-4o"


# ==============================================================================
# 2. DATENMODELLE (PYDANTIC)
# ==============================================================================
class RouterRequest(BaseModel):
    prompt: str = Field(..., description="Der Text-Prompt, der verarbeitet werden soll.", min_length=1)
    user_id: str | None = Field(default=None, description="Optionale ID des Anfragenden für Tracking oder Quotas.")


class RouterResponse(BaseModel):
    response_text: str = Field(..., description="Die generierte Antwort des gewählten LLMs.")
    model_used: str = Field(..., description="Der Name des Modells, das für die Antwort genutzt wurde.")
    latency_ms: float = Field(..., description="Die gemessene Gesamt-Latenz in Millisekunden.")


# ==============================================================================
# 3. MOCK-LLM-INFERENCE (ASYNCHRON)
# ==============================================================================
# Architektur-Hinweis: In einer Produktionsumgebung würden diese Funktionen echte HTTP-Requests
# an Provider wie OpenAI, Anthropic oder lokale vLLM-Instanzen senden.
# Hier nutzen wir asyncio.sleep(), um Netzwerklatenz und Token-Generierung zu simulieren.

async def call_fast_model(prompt: str) -> str:
    """
    Simuliert den Aufruf eines leichten, schnellen und günstigen Modells (z.B. Llama-3-8B).
    Geringe Latenz (~300ms).
    """
    logger.info(f"[Fast Model] Verarbeite Prompt (Länge: {len(prompt)} Zeichen)...")
    await asyncio.sleep(0.3)  # Simuliere 300ms Netzwerklatenz
    return f"[Fast Model Response] Einfache Antwort auf: '{prompt[:30]}...'"


async def call_heavy_model(prompt: str) -> str:
    """
    Simuliert den Aufruf eines komplexen, leistungsstarken und teureren Modells (z.B. GPT-4o / Claude 3.5 Sonnet).
    Höhere Latenz (~2.0s).
    """
    logger.info(f"[Heavy Model] Verarbeite Prompt (Länge: {len(prompt)} Zeichen)...")
    await asyncio.sleep(2.0)  # Simuliere 2000ms Netzwerklatenz
    return (
        f"[Heavy Model Response] Detaillierte und tiefe Analyse für den Prompt:\n"
        f"'{prompt}'\n"
        f"-> Ergebnisse basieren auf fortgeschrittenem Reasoning."
    )


# ==============================================================================
# 4. DIE CORE ROUTING-LOGIK / MUSS NOCH GEUPDATED WERDEN!!!
# ==============================================================================
def route_prompt(prompt: str) -> ModelType:
    """
    Entscheidet basierend auf Prompt-Eigenschaften, welches Modell verwendet werden soll.
    
    Regel:
    - Wenn der Prompt kürzer als 50 Zeichen ist ODER das Wort "zusammenfassen" enthält: 
      -> Route zum "Fast Model"  
    - Ansonsten:
      -> Route zum "Heavy Model"
    """
    prompt_lower = prompt.lower()
    
    # Heuristik 1: Prompt-Länge (< 50 Zeichen deutet auf eine kurze/einfache Anfrage hin)
    is_short = len(prompt) < 50
    
    # Heuristik 2: Schlüsselwörter (z.B. "zusammenfassen" ist oft eine Standaraufgabe)
    has_simple_keyword = "zusammenfassen" in prompt_lower
    
    if is_short or has_simple_keyword:
        reason = "Kurzer Prompt (<50 Zeichen)" if is_short else "Enthält Schlüsselwort 'zusammenfassen'"
        logger.info(f"[Routing] Wähle FAST MODEL ({ModelType.FAST.value}). Grund: {reason}")
        return ModelType.FAST
    else:
        logger.info(f"[Routing] Wähle HEAVY MODEL ({ModelType.HEAVY.value}). Grund: Komplexer/Langer Prompt ({len(prompt)} Zeichen)")
        return ModelType.HEAVY


# ==============================================================================
# 5. DER API-ENDPUNKT
# ==============================================================================
@app.post("/v1/chat/completions", response_model=RouterResponse)
async def chat_completions(request: RouterRequest) -> RouterResponse:
    """
    Haupt-Endpunkt des Routers.
    Nimmt Anfragen entgegen, bestimmt das optimale Modell, führt die Asynchrone
    Inferenz aus und misst exakt die benötigte Zeit.
    """
    start_time = time.perf_counter()
    logger.info(f"Neue Anfrage empfangen. User-ID: {request.user_id or 'anonymous'}")

    try:
        # Step 1: Modell wählen
        target_model = route_prompt(request.prompt)

        # Step 2: Modell asynchron aufrufen
        if target_model == ModelType.FAST:
            response_text = await call_fast_model(request.prompt)
        else:
            response_text = await call_heavy_model(request.prompt)

        # Step 3: Latenz messen (perf_counter liefert Sekunden als float, umgerechnet in ms)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        logger.info(f"Anfrage erfolgreich abgeschlossen in {latency_ms:.2f}ms | Modell: {target_model.value}")

        return RouterResponse(
            response_text=response_text,
            model_used=target_model.value,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Fehler bei der Prompt-Verarbeitung: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Interner Fehler beim Model-Routing.")


@app.get("/health")
async def health_check():
    """Einfacher Healthcheck-Endpunkt für Container/Monitoring."""
    return {"status": "ok", "service": "llm-router"}

