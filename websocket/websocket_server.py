import asyncio
import websockets
import json
from . import config as cf
import logging
import os
import random
import re
from collections import deque
from time import time

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('websocket.log')
    ]
)
logger = logging.getLogger(__name__)

chat_history = []
user_utterances = deque(maxlen=100)  # Store last 100 utterances
conversation_start_time = None
overlapped_speech_count = 0

# Placeholder functions for biomarker score generation
def generate_pragmatic_score(user_utt):
    return random.random()

def generate_grammar_score(user_utt):
    return random.random()

def generate_turntaking_score():
    global overlapped_speech_count
    # Normalize the overlapped speech count to a 0-1 range
    # Assuming a maximum of 10 overlaps in a conversation
    normalized_score = min(overlapped_speech_count / 10, 1)
    # Decay the score over time to prevent it from staying at 1 indefinitely
    overlapped_speech_count = max(0, overlapped_speech_count - 0.1)
    return normalized_score

def generate_anomia_score(user_utts, duration):
    def extract_fillers_and_rate(input_data, duration_seconds):
        pattern = r'\b(u+h+|a+h+|u+m+|h+m+|h+u+h+|m+h+|h+m+|h+a+h+)\b'
        all_fillers = []
        for sentence in input_data:
            filler_words = re.findall(pattern, sentence, re.IGNORECASE)
            all_fillers.extend(filler_words)
        
        duration_minutes = duration_seconds / 60
        fillers_per_minute = len(all_fillers) / duration_minutes if duration_minutes > 0 else 0
        return min(fillers_per_minute / 10, 1)  # Normalize to 0-1 range, assuming 10 fillers/min is the max

    return extract_fillers_and_rate(user_utts, duration)

def generate_prosody_score(user_utt):
    return random.random()

def generate_pronunciation_score(user_utt):
    return random.random()

def generate_biomarker_scores(user_utt):
    global user_utterances
    
    user_utterances.append(user_utt)
    
    return {
        'pragmatic': generate_pragmatic_score(user_utt),
        'grammar': generate_grammar_score(user_utt),
        'prosody': generate_prosody_score(user_utt),
        'pronunciation': generate_pronunciation_score(user_utt)
    }

async def send_periodic_scores(websocket):
    global conversation_start_time, user_utterances, overlapped_speech_count
    while True:
        await asyncio.sleep(5)  # Wait for 5 seconds
        if conversation_start_time is not None:
            current_duration = time() - conversation_start_time
            anomia_score = generate_anomia_score(list(user_utterances), current_duration)
            turntaking_score = generate_turntaking_score()
            await websocket.send(json.dumps({
                'type': 'periodic_scores',
                'data': {
                    'anomia': anomia_score,
                    'turntaking': turntaking_score
                }
            }))

async def handle_client(websocket):
    try:
        # Add CORS headers
        websocket.request_headers["Access-Control-Allow-Origin"] = "*"
        websocket.request_headers["Access-Control-Allow-Methods"] = "GET, POST"
        websocket.request_headers["Access-Control-Allow-Headers"] = "*"
        
        global conversation_start_time, overlapped_speech_count
        client_id = id(websocket)
        client_address = websocket.remote_address
        logger.info(f"New client connected: {client_id} from {client_address}")
        
        conversation_start_time = time()
        periodic_scores_task = asyncio.create_task(send_periodic_scores(websocket))
        
        try:
            async for message in websocket:
                logger.info(f"Received message from client {client_id}: {message}")
                data = json.loads(message)
                
                if data['type'] == 'overlapped_speech':
                    overlapped_speech_count += 1
                    logger.info(f"Overlapped speech detected. Count: {overlapped_speech_count}")
                
                elif data['type'] == 'transcription':
                    user_utt = data['data'].lower()
                    logger.info(f"Received user utterance from client {client_id}: {user_utt}")
                    
                    # Generate biomarker scores
                    biomarker_scores = generate_biomarker_scores(user_utt)
                    logger.info(f"Generated biomarker scores for client {client_id}: {biomarker_scores}")
                    await websocket.send(json.dumps({'type': 'biomarker_scores', 'data': biomarker_scores}))
                    
                    # Generate LLM response
                    response = process_user_utterance(user_utt)
                    logger.info(f"Generated response for client {client_id}: {response}")
                    await websocket.send(json.dumps({'type': 'llm_response', 'data': response}))

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed for client {client_id}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for client {client_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            periodic_scores_task.cancel()
            conversation_start_time = None
            user_utterances.clear()
            overlapped_speech_count = 0
            logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in handle_client: {e}")

def process_user_utterance(user_utt):
    global chat_history
    
    try:
        # Prepare input for LLM
        history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        
        input_text = f"<|system|>\n{cf.prompt}<|end|>"
        for turn in history:
            if turn['Speaker'] == 'User':
                input_text += f"\n<|user|>\n{turn['Utt']}<|end|>"
            else:
                input_text += f"\n<|assistant|>\n{turn['Utt']}<|end|>"
        input_text += f"\n<|user|>\n{user_utt}<|end|>\n<|assistant|>\n"
        
        logger.debug(f"Input text for LLM: {input_text}")
        
        # Generate response using LLM
        output = cf.llm(input_text, max_tokens=cf.max_length, stop=["<|end|>",".", "?"], echo=True)
        system_utt = (output['choices'][0]['text'].split("<|assistant|>")[-1]).strip()
        
        # Update chat history
        chat_history.append({'Speaker': 'User', 'Utt': user_utt})
        chat_history.append({'Speaker': 'System', 'Utt': system_utt})
        
        return system_utt
    except Exception as e:
        logger.error(f"Error in process_user_utterance: {e}")
        return "I'm sorry, I encountered an error while processing your request."

async def main():
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8765))
    logger.info("Starting WebSocket server...")
    
    try:
        server = await websockets.serve(
            handle_client, 
            host,
            port,
            ping_interval=None  # Add this to prevent connection timeouts
        )
        logger.info(f"WebSocket server started successfully on ws://{host}:{port}")
        await server.wait_closed()
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server crashed: {e}")