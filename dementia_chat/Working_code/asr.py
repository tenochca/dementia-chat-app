import sys
import keyboard
import re
import azure.cognitiveservices.speech as speechsdk
import csv
import time
import datetime


from Working_code import config as cf
from Working_code import tts

# set API keys
speech_key, service_region = cf.speech_key, cf.service_region
speech_config = cf.speech_config
audio_config = cf.audio_config

# set logger
logger = cf.logging.getLogger("__asr__")


def record_chat():
    mode = "w" if cf.script_check == 1 else "a"
    with open(cf.script_path, encoding="utf-8-sig", newline='', mode=mode) as f:
        writer = csv.writer(f)
        cf.chat_history.sort(key=lambda x: x['Time'])

        if cf.script_check == 1:
            writer.writerow(['Speaker', 'Utt', 'Time'])
        writer.writerow(cf.chat_history[-1].values())
    cf.script_check += 1


# ASR function starts
class listen_micr:

    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def run(self):
        # Set up recognizer (Using MS Azure)
        # Sample codes could be checked
        # https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/quickstart/python/from-microphone/quickstart.py
        # https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/a5de28baa82f2633d38e2acd49a319b9df2104c3/samples/python/console/speech_sample.py#L225
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=cf.THIS_LANGUAGE, audio_config=audio_config)

        while self._running:
            try:
                record_ongoing = speech_recognizer.recognize_once_async()
                logger.debug('listening ...')
                
                # Inform a person to know start of recognition
                print("Say Something!")

                # Recognizing speech through mike / designated speaker
                result = record_ongoing.get()
                
                # ignore punctuation marks and lower all words
                # ex) I eat foods. -> I eat foods (ignore punctuation) -> i eat foods (lower words)
                user_utt = result.text[:-1].lower()
                
                if user_utt:
                    utt_start_time = round(time.time() - cf.game_start_time,5)
                    
                    # print the result to see how the recognizer recognizes
                    print("Recognized: {}".format(user_utt))
                    logger.info('user said: ' + user_utt)
                    self.respond_to_user_utt(user_utt, cf.chat_history)

                    user_utt = re.sub(r'[^a-zA-Z ]', '', user_utt).lower()
                    start_time = str(datetime.timedelta(seconds=utt_start_time)).split(".")[0]
                    
                    one_turn_history = {'Speaker': 'User', 'Utt': user_utt, 'Time': start_time}
                    cf.chat_history.append(one_turn_history)

                    record_chat()

                # Force to shut down ASR only
                # pressing ctrl + z shut down ASR
                if ('exit' in user_utt) or ('종료' in user_utt) or (keyboard.is_pressed('esc')):
                    print("Exiting...")
                    sys.exit()

            # If error occurs, pass and do the recognition task again
            # MS Azure recognizer errors are written below as exception
            except result.reason == speechsdk.ResultReason.NoMatch:
                logger.info("No speech could be recognized: {}".format(result.no_match_details))
                pass
            except result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.info("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.info("Error details: {}".format(cancellation_details.error_details))
                pass

    # Response selection based on the ASR result
    def respond_to_user_utt(self, user_utt, chat_history):
        try:
            history = []
            for i in range(max(0, len(chat_history) - 6), len(chat_history)):
                history.append(chat_history[i])

            logger.info("history: " + str(history))
            input_text = f"<|system|>\n{cf.prompt}<|end|>"
            for turn in history:
                if turn['Speaker'] == 'User':
                    input_text += f"\n<|user|>\n{turn['Utt']}<|end|>"
                else:
                    input_text += f"\n<|assistant|>\n{turn['Utt']}<|end|>"
            input_text += f"\n<|user|>\n{user_utt}<|end|>\n<|assistant|>\n"
            logger.info("input_text: " + input_text)
            
            output = cf.llm(input_text, max_tokens=cf.max_length, stop=["<|end|>", ".", "?"], echo=True)

            system_utt = (output['choices'][0]['text'].split("<|assistant|>")[-1]).strip()
            tts.synthesize_utt(system_utt)
        # If error occurs, write down the error at the logs/dm.log file
        except Exception as err:
            logger.error("user input parsing failed " + str(err))