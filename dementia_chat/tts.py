# -*- coding:utf-8 -*-
'''
Synthesize utterances using Microsoft Azure TTS SDK mainly from 'parse_tree.py' file and 'asr.py' file.
'''
import azure.cognitiveservices.speech as speechsdk
import time
import datetime
import re

from . import asr
from . import config as cf

# set API keys
speech_key, service_region = cf.speech_key, cf.service_region
speech_config = cf.speech_config

# set logger
logger = cf.logging.getLogger("__tts__")

# MS Azure Text to Speech(TTS) SDK synthesize the sentence
def synthesize_utt(utterance):
    # get a text and synthesize it
    
    if cf.overlap_check == 0 and utterance != None:
        cf.overlap_check = 1
        utt_start_time = round(time.time() - cf.game_start_time,5)
        logger.info(f"New utterance is: {utterance}")
        
        # MS Azure TTS / Synthesize text and make it to speak
        # Sample codes could be checked right below link
        # https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/quickstart/python/text-to-speech/quickstart.py
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # print result which utterance is selected
        print()
        print('**********' * 5)
        print('System: ', utterance)
        print('**********' * 5)
        print()
        
        start_time = str(datetime.timedelta(seconds=utt_start_time)).split(".")[0]

        one_turn_history = {'Speaker': 'System', 'Utt': utterance, 'Time': start_time}
        cf.chat_history.append(one_turn_history)

        asr.record_chat()
        
        result = speech_synthesizer.speak_text_async(utterance).get()
        utterance = re.sub(r'[^a-zA-Z ]', '', utterance).lower()

        # If error occurs, print it and pass to work next utterance with no problem
        # Below is an error that is occurring for MS Azure
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.info("Error details: {}".format(cancellation_details.error_details))
        
        # ToDo: put synthesizer and player in separate threads and queue
        time.sleep(0.1)
        cf.overlap_check = 0