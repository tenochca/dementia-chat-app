import base64
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from time import time
import logging
import random
import re
from collections import deque
import asyncio
from .. import config as cf

logger = logging.getLogger(__name__)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.client_id = id(self)
        try:
            # Verify LLM is initialized
            if not cf.llm:
                raise RuntimeError("LLM not initialized")
            # Rest of connect code...
            self.conversation_start_time = time()
            self.user_utterances = deque(maxlen=100)
            self.overlapped_speech_count = 0
            self.chat_history = []  # Add chat_history as instance variable
            await self.accept()
            self.periodic_scores_task = asyncio.create_task(self.send_periodic_scores())
        except Exception as e:
            logger.error(f"Failed to initialize consumer: {e}")
            return

    async def disconnect(self, close_code):
        if hasattr(self, 'periodic_scores_task'):
            self.periodic_scores_task.cancel()
        self.conversation_start_time = None
        self.user_utterances.clear()
        self.overlapped_speech_count = 0
        logger.info(f"Client disconnected: {self.client_id}")

    def process_user_utterance(self, user_utt):
        try:
            # Prepare input for LLM
            history = self.chat_history[-5:] if len(self.chat_history) > 5 else self.chat_history
            
            input_text = f"<|system|>\n{cf.prompt}<|end|>"
            for turn in history:
                if turn['Speaker'] == 'User':
                    input_text += f"\n<|user|>\n{turn['Utt']}<|end|>"
                else:
                    input_text += f"\n<|assistant|>\n{turn['Utt']}<|end|>"
            input_text += f"\n<|user|>\n{user_utt}<|end|>\n<|assistant|>\n"
            
            # Generate response using LLM
            output = cf.llm(input_text, max_tokens=cf.max_length, stop=["<|end|>",".", "?"], echo=True)
            system_utt = (output['choices'][0]['text'].split("<|assistant|>")[-1]).strip()
            
            # Update chat history
            self.chat_history.append({'Speaker': 'User', 'Utt': user_utt})
            self.chat_history.append({'Speaker': 'System', 'Utt': system_utt})
            
            return system_utt
        except Exception as e:
            logger.error(f"Error in process_user_utterance: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            
            if data['type'] == 'overlapped_speech':
                self.overlapped_speech_count += 1
                logger.info(f"Overlapped speech detected. Count: {self.overlapped_speech_count}")
            
            elif data['type'] == 'transcription':
                user_utt = data['data'].lower()
                logger.info(f"Received user utterance: {user_utt}")
                
                # Generate biomarker scores
                biomarker_scores = self.generate_biomarker_scores(user_utt)
                await self.send(json.dumps({
                    'type': 'biomarker_scores',
                    'data': biomarker_scores
                }))
                
                # Generate LLM response
                response = self.process_user_utterance(user_utt)
                await self.send(json.dumps({
                    'type': 'llm_response',
                    'data': response
                }))
            
            elif data['type'] == 'audio_data':
                await self.process_audio_data(data['data'], data['sampleRate'])
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")

    async def process_audio_data(self, base64_data, sample_rate):
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(base64_data)
            
            # Process the audio bytes as needed
            # For example, you can save the audio to a file or send it to a speech recognition service
            
            logger.info(f"Received audio data: {len(audio_bytes)} bytes at {sample_rate}Hz")
            
            # Example: Save audio to a file (optional)
            with open(f'audio_{self.client_id}.wav', 'ab') as f:
                f.write(audio_bytes)
            
            # Example: Send audio to a speech recognition service (optional)
            # transcription = await speech_recognition_service(audio_bytes, sample_rate)
            # logger.info(f"Transcription: {transcription}")
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

    def generate_pragmatic_score(self, user_utt):
        return random.random()

    def generate_grammar_score(self, user_utt):
        return random.random()

    def generate_turntaking_score(self):
        normalized_score = min(self.overlapped_speech_count / 10, 1)
        self.overlapped_speech_count = max(0, self.overlapped_speech_count - 0.1)
        return normalized_score

    def generate_anomia_score(self):
        pattern = r'\b(u+h+|a+h+|u+m+|h+m+|h+u+h+|m+h+|h+m+|h+a+h+)\b'
        all_fillers = []
        for sentence in self.user_utterances:
            filler_words = re.findall(pattern, sentence, re.IGNORECASE)
            all_fillers.extend(filler_words)
        
        duration_minutes = (time() - self.conversation_start_time) / 60
        fillers_per_minute = len(all_fillers) / duration_minutes if duration_minutes > 0 else 0
        return min(fillers_per_minute / 10, 1)

    def generate_prosody_score(self, user_utt):
        return random.random()

    def generate_pronunciation_score(self, user_utt):
        return random.random()

    def generate_biomarker_scores(self, user_utt):
        self.user_utterances.append(user_utt)
        return {
            'pragmatic': self.generate_pragmatic_score(user_utt),
            'grammar': self.generate_grammar_score(user_utt),
            'prosody': self.generate_prosody_score(user_utt),
            'pronunciation': self.generate_pronunciation_score(user_utt)
        }
    
    async def send_periodic_scores(self):  # Remove websocket parameter
        while True:
            await asyncio.sleep(5)
            if self.conversation_start_time is not None:
                current_duration = time() - self.conversation_start_time
                anomia_score = self.generate_anomia_score()
                turntaking_score = self.generate_turntaking_score()
                await self.send(json.dumps({  # Use self.send instead
                    'type': 'periodic_scores',
                    'data': {
                        'anomia': anomia_score,
                        'turntaking': turntaking_score
                    }
                }))