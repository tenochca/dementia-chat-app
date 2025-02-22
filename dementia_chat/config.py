######################################################### load packages

# generaL behavior
import os
import time
from llama_cpp import Llama

# for logging:
import warnings, logging

#for TTS
import azure.cognitiveservices.speech as speechsdk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

######################################################### set API keys

# MS Auzre / used at 'tts.py' and 'asr.py' files
speech_key, service_region = "3249fb4e6d8248569b42d5dbf693c259", "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# audio_config = speechsdk.audio.AudioConfig(device_name="{0.0.1.00000000}.{9485502f-1e25-43a1-b32e-f2064ed250be}")
# audio_config = speechsdk.audio.AudioConfig(device_name="{0.0.1.00000000}.{c600777f-5cb7-44a2-9457-68fe97eb7632}")

audio_device_name = os.getenv("AUDIO_DEVICE_NAME", None)
if audio_device_name:
    audio_config = speechsdk.audio.AudioConfig(device_name=audio_device_name)
else:
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

######################################################### set logging

# Ignoring the warnings
warnings.filterwarnings(action='ignore')

# Making the 'logs' folder if the folder does not exist
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

if not os.path.exists("./script/"):
    os.mkdir("./script/")

# Set up log file written format ex) 01:39:09
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    level=logging.DEBUG,
    filename='./logs/dm.log',
    # encoding='utf-8',
    datefmt="%H:%M:%S",
    #stream=sys.stderr,
)

logging.getLogger("chardet.charsetprober").disabled = True
logger = logging.getLogger(__name__)

######################################################### set asr

script_check = 1
chat_history = list()
THIS_LANGUAGE = 'en-US'
game_start_time = time.time()
overlap_check = 0
script_path = './Script/Script' +"("+time.strftime('%y-%m-%d %H-%M', time.localtime(time.time()))+")"+'.csv'

######################################################### set global variables

voice = f"Microsoft Server Speech Text to Speech Voice ({THIS_LANGUAGE}, JennyNeural)"
speech_config.speech_synthesis_voice_name = voice

######################################################### set llm settings

current_path = os.path.dirname(os.path.abspath(__file__))

max_length = 256
prompt = "You are an assistant for dementia patients. Provide any response as much short as possible."

try:
    model_path = current_path + "/services/Phi-3_finetuned.gguf"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    llm = Llama(
        model_path=model_path,
        n_ctx=max_length,
        n_threads=16,
        n_gpu_layers=0
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

######### PROSODY & PRONUNCIATION MODEL PATHS #############
pronunciation_model_path = "dementia_chat\services\pronunciation_rf(v4).pkl"
prosody_model_path = "dementia_chat\services\prosody_rf(v4).pkl"